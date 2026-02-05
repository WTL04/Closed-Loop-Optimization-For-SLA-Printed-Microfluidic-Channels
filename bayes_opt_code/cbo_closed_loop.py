#!/usr/bin/env python3
"""
cbo_closed_loop.py
----------------------------------------
Online / Closed-Loop Optimization
- Observe new context
- Suggest next set of print parameters
- Run experiment (simulate or real)
- Update dataset
- Retrain surrogate model
- Runs for a fixed number of iterations unless stopped manually
- Visualize improvement across iterations
"""

import time
import numpy as np
import pandas as pd
import os
from sheets_logger import GoogleSheetsLogger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from cbo import ContextualBayesOpt
from visualize import visualize_model_convergence
from sklearn.metrics import r2_score
from visualize import visualize_control_chart
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

channels_per_batch = 13
debug = False

# Load & merge datasets
def load_dataset_csv():
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "datasets" / "dataset.csv"
    return pd.read_csv(csv_path)

def choose_training_source():
    print("\nChoose training data source:")
    print("1) Local CSV (datasets/dataset.csv)")
    print("2) Google Sheet (read existing lab data)")
    choice = input("Enter 1 or 2 [default = 1]: ").strip()
    return "sheet" if choice == "2" else "csv"

def choose_dataset_path():
    """
    If only one CSV exists in datasets/, use it automatically.
    If multiple CSVs exist, prompt the user to choose.
    """
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root / "datasets"
    csvs = sorted(datasets_dir.glob("*.csv"))

    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {datasets_dir}")

    # If only one CSV, auto-select it
    if len(csvs) == 1:
        print(f"[Data] Using dataset: {csvs[0].name}")
        return csvs[0]

    # If multiple CSVs, ask the user
    print("\nSelect dataset CSV to train on:")
    for idx, p in enumerate(csvs, start=1):
        print(f"  {idx}) {p.name}")

    while True:
        choice = input(f"Enter choice [1-{len(csvs)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(csvs):
            return csvs[int(choice) - 1]
        print("Invalid selection. Please enter a valid number.")

def get_number_of_runs(default=1):
    val = input(f"Enter number of independent runs [default = {default}]: ").strip()
    if val == "":
        return default
    return int(val)

def get_initial_context_snapshot():
    print("1) Manually input context snapshot  2) Use fixed testing context snapshot")
    choice = input("Choose 1 or 2 [default = 2]: ").strip()
    if choice == "1":
        ambient_temp = float(input("ambient_temp (°F): ").strip())
        resin_temp = float(input("resin_temp (°F): ").strip())
        resin_age = float(input("resin_age (days): ").strip())
    else:
        ambient_temp = 80.0
        resin_temp = 80.0
        resin_age = 15.0
    return {
        "ambient_temp": ambient_temp,
        "resin_temp": resin_temp,
        "resin_age": resin_age,
    }

# Simulate (or run) a new print batch
def run_experiment(params, context, batch_id, channels_per_batch, sheet_manual_entry=False, simulate=True):
    """
    Simulates one batch print (channels_per_batch channels).
    Synthetic model is designed to mimic real data behavior:
      - Mean flow depends on knobs + context.
      - Variability has TWO components:
          (1) Batch-level drift (common bias across all channels in the batch)
          (2) Channel-level noise (per-channel variation)
    Later, set simulate=False to replace with real measured flows.
    """

    # 1) Mean flow model 
    mean_flow = 100.0  # baseline nominal flow rate (mL/min)

    # layer thickness decoded to 50 or 100 
    lt_value = params["layer_thickness_um"]  # 50 or 100
    lt_factor = 1.10 if lt_value == 50 else 0.90
    mean_flow *= lt_factor

    # orientation / z rotation effect 
    ori = float(params.get("z_rotation_deg", 45.0))
    ori_factor = 1.0 - abs(ori - 45.0) / 200.0
    mean_flow *= max(0.10, ori_factor)  # keep positive

    # fit adjustment 
    fit = float(params.get("fit_adjustment", 0.0))
    fit_factor = 1.0 - abs(fit) / 3000.0
    mean_flow *= max(0.10, fit_factor)

    # resin temperature effect 
    temp_factor = 1.0 + (float(context["resin_temp"]) - 72.0) * 0.01
    mean_flow *= max(0.10, temp_factor)

    # ambient temperature effect 
    ambient_factor = 1.0 + (float(context["ambient_temp"]) - 72.0) * 0.005
    mean_flow *= max(0.10, ambient_factor)

    # resin aging effect 
    age_factor = 1.0 - float(context["resin_age"]) * 0.002
    mean_flow *= max(0.10, age_factor)

    # 2) Variability model 
    # fit here is already decoded (-250..250), so normalize by 250
    p_ori = abs(ori - 45.0) / 45.0            # 0 at 45deg, ~1 at 0/90
    p_fit = min(abs(fit) / 250.0, 1.0)        # 0 at 0, 1 at +/-250
    p_lt = 0.15 if lt_value == 100 else 0.0   

    # Context penalties (drift increases variability)
    p_age = min(float(context["resin_age"]) / 30.0, 1.0)
    p_temp = min(abs(float(context["resin_temp"]) - 72.0) / 10.0, 1.0)

    # Channel-level noise fraction 
    channel_noise_frac = (
        0.006
        + 0.010 * p_ori
        + 0.008 * p_fit
        + 0.004 * p_lt
        + 0.006 * p_age
        + 0.004 * p_temp
    )
    channel_noise_frac = float(np.clip(channel_noise_frac, 0.002, 0.06))

    # Batch-level drift fraction
    batch_drift_frac = (
        0.002
        + 0.003 * p_age
        + 0.002 * p_temp
        + 0.002 * p_ori
        + 0.001 * p_fit
        + 0.001 * p_lt
    )
    batch_drift_frac = float(np.clip(batch_drift_frac, 0.001, 0.02))

    # 3) Generate batch data
    rows = []

    if sheet_manual_entry:
        for i in range(channels_per_batch):
            rows.append(
                {
                    "batch_id": int(batch_id),
                    "channel_id": int(i + 1),
                    "layer_thickness_um": lt_value,
                    "z_rotation_deg": ori,
                    "fit_adjustment": fit,
                    "resin_age": float(context["resin_age"]),
                    "resin_temp": float(context["resin_temp"]),
                    "ambient_temp": float(context["ambient_temp"]),
                    "channel_flow_rate_ml_per_min": "",  # leave blank for manual entry
                }
            )
        return pd.DataFrame(rows)

    if simulate:
        # Common drift term applies to all channels in the batch
        batch_bias = np.random.normal(0.0, batch_drift_frac * mean_flow)
    else:
        batch_bias = 0.0

    for i in range(channels_per_batch):
        if simulate:
            channel_noise = np.random.normal(0.0, channel_noise_frac * mean_flow)
            flow = max(mean_flow + batch_bias + channel_noise, 1e-6)
        else:
            while True:
                try:
                    v = input(f"Measured flow for channel {i + 1}: ").strip()
                    flow = float(v)
                    break
                except ValueError:
                    print("Invalid input. Enter numeric flow.")

        rows.append(
            {
                "batch_id": int(batch_id),
                "channel_id": int(i + 1),
                "layer_thickness_um": lt_value,
                "z_rotation_deg": ori,
                "fit_adjustment": fit,
                "resin_age": float(context["resin_age"]),
                "resin_temp": float(context["resin_temp"]),
                "ambient_temp": float(context["ambient_temp"]),
                "channel_flow_rate_ml_per_min": flow,
            }
    )
    return pd.DataFrame(rows)

# Compute CV & prep training data
def update_training_data(df_all):
    """
    Computes CV per batch using ONLY rows that have measured flow values.
    Ignores incomplete batches (flow NaN) until the user fills them in.
    Returns: X, y, df_batches
    """
    flow_col = "channel_flow_rate_ml_per_min"
    df_all = df_all.copy()
    # ensure column exists
    if flow_col not in df_all.columns:
        df_all[flow_col] = pd.NA

    df_all[flow_col] = pd.to_numeric(df_all[flow_col], errors="coerce")
    df_valid = df_all.dropna(subset=[flow_col]).copy()

    if df_valid.empty:
        raise ValueError("No completed flow measurements found. Please enter flows in the dataset or sheet.")

    summary = (
        df_valid.groupby("batch_id")[flow_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["cv"] = summary["std"] / summary["mean"]
    summary["delta_cv"] = summary["cv"].diff().fillna(0)

    features = [
        "layer_thickness_um",
        "z_rotation_deg",
        "fit_adjustment",
        "resin_temp",
        "ambient_temp",
        "resin_age",
    ]

    df_batches = (
        df_valid.groupby("batch_id")
        .first()[features]
        .reset_index()
        .merge(summary[["batch_id", "cv", "delta_cv"]], on="batch_id")
    )

    # encode thickness to 0/1
    df_batches["layer_thickness_um"] = df_batches["layer_thickness_um"].replace({50: 0, 100: 1})

    fit_map = {v: i for i, v in enumerate([-250, -150, -50, 0, 50, 150, 250])}
    df_batches["fit_adjustment"] = df_batches["fit_adjustment"].map(fit_map)

    X, y = df_batches[features], df_batches["cv"]
    return X, y, df_batches

# Main Closed-Loop Function 
def main(num_runs=1, max_iterations=10, tolerance=0.005):
    # Prompts the user for the number of independent runs
    num_runs = get_number_of_runs(default=num_runs)
    print(f"Running {num_runs} independent runs")

    # Initializes Google Sheets logging if credentials are available
    gs_logger = None
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    sheet_id = os.environ.get("SHEET_ID")
    worksheet = os.environ.get("WORKSHEET_NAME", "Sheet2")

    if creds_path and sheet_id:
        try:
            gs_logger = GoogleSheetsLogger(creds_path, sheet_id, worksheet)
            print(f"[Sheets] Logging enabled -> tab '{worksheet}'")
        except Exception as e:
            print("[Sheets] Logging init failed:", e)
            gs_logger = None
    else:
        print("[Sheets] Logging disabled")

    # Prompts the user to select the training data source
    training_source = choose_training_source()
    print(f"[Data] Training source: {training_source}")

    # if CSV and you want choose path:
    if training_source == "csv":
        dataset_csv = choose_dataset_path()
        print(f"[Data] Using CSV: {dataset_csv}")

    all_histories = []

    for run in range(num_runs):
        print(f"\n========== RUN {run + 1}/{num_runs} ==========")

        base_context = get_initial_context_snapshot()
        print("Initial context snapshot:", base_context)
        c_new = dict(base_context)

        features = [
            "layer_thickness_um",
            "z_rotation_deg",
            "fit_adjustment",
            "resin_temp",
            "ambient_temp",
            "resin_age",
        ]
        categorical = ["layer_thickness_um", "fit_adjustment"]
        numerical = [f for f in features if f not in categorical]

        preprocess = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
                ("num", StandardScaler(), numerical),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocess", preprocess),
                ("rf", RandomForestRegressor(random_state=42 + run)),
            ]
        )

        pbounds = {
            "layer_thickness_um": (0, 1),  # encoded 0/1
            "z_rotation_deg": (0, 90),
            "fit_adjustment": (0, 6),  # encoded 0..6
        }

        cbo = ContextualBayesOpt(pipeline=pipeline, pbounds=pbounds)

        # Train on initial dataset
        if training_source == "sheet":
            if gs_logger is None:
                raise RuntimeError("Sheets selected but logger not initialized.")
            df_hist = gs_logger.read_dataframe()
        else:
            df_hist = pd.read_csv(dataset_csv)

        X, y, df_batches = update_training_data(df_hist)
        cbo.train_surrogate(X, y, verbose=False)

        print(f"Loaded dataset with {len(df_batches)} batches.")

        if debug:
            r2 = r2_score(y, cbo.model.predict(X))
            print(f"[DEBUG] Initial R2_train = {r2:.3f}")

        prev_cv = float(df_batches["cv"].iloc[-1])
        cv_history = [prev_cv]

        # determine starting batch id (next available integer)
        try:
            start_batch_id = int(pd.to_numeric(df_hist["batch_id"], errors="coerce").max()) + 1
        except Exception:
            start_batch_id = 1

        for i in range(1, max_iterations + 1):
            print(f"\n--- Iteration {i} ---")

            # context drift (if desired)
            c_new["ambient_temp"] -= 0.2
            c_new["resin_temp"] -= 0.1
            c_new["resin_age"] += 0.2
            print("Context:", c_new)

            # get suggestion from BO
            best_params, best_lcb, _ = cbo.compute_bayes_opt(c_new, verbose=True)

            # Decodes discrete variables from optimizer encoding into physical print settings
            best_params["layer_thickness_um"] = 50 if int(round(best_params["layer_thickness_um"])) == 0 else 100
            fit_vals = [-250, -150, -50, 0, 50, 150, 250]
            best_params["fit_adjustment"] = fit_vals[int(best_params["fit_adjustment"])]

            print("Suggested parameters:", best_params)
            print(f"Best LCB (surrogate score): {best_lcb:.6f}")

            clean_batch_id = start_batch_id + (i - 1)

            # create template rows (13 rows with blank flow) and append to sheet
            df_template = run_experiment(
                best_params,
                c_new,
                batch_id=clean_batch_id,
                channels_per_batch=channels_per_batch,
                simulate=True,  # keep simulate True so context->mean calc still used (not necessary but ok)
                sheet_manual_entry=True
            )

            if gs_logger is not None:
                gs_logger.append_dataframe(df_template)
                print(f"[Sheets] Appended template for batch_id = {clean_batch_id} (please fill flows manually)")

            while True:
                done = input("Have the prints been completed? (y/n): ").strip().lower()
                if done not in ("y", "n"):
                    print("Please enter 'y' or 'n'.")
                    continue
                if done == "n":
                    print("Waiting for prints to complete. Re-check when ready.")
                    time.sleep(2)
                    continue

                entered = input("Have all channel flow rates been entered manually? (y/n): ").strip().lower()
                if entered not in ("y", "n"):
                    print("Please enter 'y' or 'n'.")
                    continue
                if entered == "n":
                    print("Please enter all channel flow rates before proceeding.")
                    time.sleep(2)
                    continue

                # Only reach here if both answers are 'y'
                break

            # reload dataset (from sheet or csv)
            if training_source == "sheet":
                df_hist = gs_logger.read_dataframe()
            else:
                df_hist = pd.read_csv(dataset_csv)

            # update training dataset (ignores incomplete batches)
            try:
                X, y, df_batches = update_training_data(df_hist)
            except ValueError as e:
                print("[TRAINING] Incomplete data:", e)
                print("Please ensure flows are entered for the new batch and try again.")
                break

            cbo.train_surrogate(X, y, verbose=False)

            if debug:
                r2 = r2_score(y, cbo.model.predict(X))
                print(f"[DEBUG] Iter {i}: R2_train = {r2:.3f}")

            new_cv = float(df_batches["cv"].iloc[-1])
            delta_cv = new_cv - prev_cv
            cv_history.append(new_cv)
            print(f"Prev CV: {prev_cv:.4f}, New CV: {new_cv:.4f}, delta: {delta_cv:.4f}")
            prev_cv = new_cv

        all_histories.append(cv_history)

    visualize_model_convergence(all_histories)
    # optional control chart
    try:
        visualize_control_chart(df_batches)
    except Exception:
        pass

if __name__ == "__main__":
    main()