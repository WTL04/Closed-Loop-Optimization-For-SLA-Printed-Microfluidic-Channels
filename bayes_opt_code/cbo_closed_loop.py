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
- Stop automatically when CV stabilizes
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

# ==========================================================
#          Load & merge datasets
# ==========================================================
from pathlib import Path
import pandas as pd

def load_dataset():
    """Loads dataset.csv from ../datasets and normalizes key columns."""
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "datasets" / "dataset.csv"

    df = pd.read_csv(csv_path)

    # --- Normalize column names if you ever changed them ---
    # (only needed if your CSV sometimes uses different names)
    rename_map = {
        "channel_flow_rate_ml_per_min": "channel_flow_rate_ml_per_min",
        "channel_flow_rate_per_min": "channel_flow_rate_ml_per_min",  # old name fallback
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # --- Force batch_id numeric if possible ---
    if "batch_id" in df.columns:
        df["batch_id"] = pd.to_numeric(df["batch_id"], errors="coerce")

    # Optional: drop rows that don’t have a usable batch_id
    df = df.dropna(subset=["batch_id"]).copy()
    df["batch_id"] = df["batch_id"].astype(int)

    return df


# ==========================================================
#          Simulate (or run) a new print batch
# ==========================================================
def run_experiment(params, context, batch_id, channels_per_batch=10, simulate=True):
    """
    Simulates one batch print (channels_per_batch channels).
    Synthetic model is designed to mimic real data behavior:
      - Mean flow depends on knobs + context.
      - Variability has TWO components:
          (1) Batch-level drift (common bias across all channels in the batch)
          (2) Channel-level noise (per-channel variation)
    Later, set simulate=False to replace with real measured flows.
    """
    import time
    import numpy as np
    import pandas as pd

    # ----------------------------
    # 1) Mean flow model (systematic effects)
    # ----------------------------
    mean_flow = 100.0  # baseline nominal flow rate (mL/min, arbitrary)

    # layer thickness decoded to 50 or 100 (as you do in main)
    lt_value = params["layer_thickness_um"]  # 50 or 100
    lt_factor = 1.10 if lt_value == 50 else 0.90
    mean_flow *= lt_factor

    # orientation / z rotation effect (best near 45 deg)
    ori = float(params.get("z_rotation_deg", 45.0))
    ori_factor = 1.0 - abs(ori - 45.0) / 200.0
    mean_flow *= max(0.10, ori_factor)  # keep positive

    # fit adjustment (tight fit -> smaller flow); your fit is decoded to [-250..250]
    fit = float(params.get("fit_adjustment", 0.0))
    fit_factor = 1.0 - abs(fit) / 3000.0
    mean_flow *= max(0.10, fit_factor)

    # resin temperature effect (nominal 72F)
    temp_factor = 1.0 + (float(context["resin_temp"]) - 72.0) * 0.01
    mean_flow *= max(0.10, temp_factor)

    # ambient temperature effect (nominal 72F)
    ambient_factor = 1.0 + (float(context["ambient_temp"]) - 72.0) * 0.005
    mean_flow *= max(0.10, ambient_factor)

    # resin aging effect (older resin -> lower mean flow)
    age_factor = 1.0 - float(context["resin_age"]) * 0.002
    mean_flow *= max(0.10, age_factor)

    # ----------------------------
    # 2) Variability model (what drives CV)
    # ----------------------------
    # Penalties: farther from sweet spot => higher variance
    # Note: fit here is already decoded (-250..250), so normalize by 250
    p_ori = abs(ori - 45.0) / 45.0            # 0 at 45deg, ~1 at 0/90
    p_fit = min(abs(fit) / 250.0, 1.0)        # 0 at 0, 1 at +/-250
    p_lt = 0.15 if lt_value == 100 else 0.0   # assume 100um slightly less stable

    # Context penalties (drift increases variability)
    p_age = min(float(context["resin_age"]) / 30.0, 1.0)
    p_temp = min(abs(float(context["resin_temp"]) - 72.0) / 10.0, 1.0)

    # Channel-level noise fraction (dominant term for within-batch CV)
    # Lower is better. Tuned so CV is realistically in a small range.
    channel_noise_frac = (
        0.006
        + 0.010 * p_ori
        + 0.008 * p_fit
        + 0.004 * p_lt
        + 0.006 * p_age
        + 0.004 * p_temp
    )
    channel_noise_frac = float(np.clip(channel_noise_frac, 0.002, 0.06))

    # Batch-level drift fraction (common-mode bias across channels in the batch)
    # This captures "printer state" and environmental drift: you want BO to reduce sensitivity to it.
    batch_drift_frac = (
        0.002
        + 0.003 * p_age
        + 0.002 * p_temp
        + 0.002 * p_ori
        + 0.001 * p_fit
        + 0.001 * p_lt
    )
    batch_drift_frac = float(np.clip(batch_drift_frac, 0.001, 0.02))

    # ----------------------------
    # 3) Generate batch data
    # ----------------------------
    rows = []

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
            flow = float(input(f"Measured flow for channel {i + 1}: "))

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
            "channel_flow_rate_ml_per_min": float(flow),
        }
    )

    return pd.DataFrame(rows)


# ==========================================================
#             Compute CV & prep training data
# ==========================================================
def update_training_data(df_all):
    """Computes CV per batch, returns updated training data"""
    summary = (
        df_all.groupby("batch_id")["channel_flow_rate_ml_per_min"]
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
        df_all.groupby("batch_id")
        .first()[features]
        .reset_index()
        .merge(summary[["batch_id", "cv", "delta_cv"]], on="batch_id")
    )
    df_batches["layer_thickness_um"] = df_batches["layer_thickness_um"].replace({50: 0, 100: 1})

    # Encode fit adjustment as categorical index
    fit_map = {v: i for i, v in enumerate([-250, -150, -50, 0, 50, 150, 250])}
    df_batches["fit_adjustment"] = df_batches["fit_adjustment"].map(fit_map)

    X, y = df_batches[features], df_batches["cv"]
    return X, y, df_batches
def get_number_of_runs(default=3):
    val = input(f"Enter number of independent runs [default={default}]: ").strip()
    if val == "":
        return default
    return int(val)

def get_initial_context_snapshot():
    print("1) Manually input context snapshot  2) Use fixed testing context snapshot: ", end="")
    choice = input().strip()

    if choice == "1":
        ambient_temp = float(input("ambient_temp (°F): ").strip())
        resin_temp = float(input("resin_temp (°F): ").strip())
        resin_age = float(input("resin_age (estimated hours since opened): ").strip())
    else:
        # Fixed testing snapshot (edit these defaults if you want)
        ambient_temp = 75.0
        resin_temp = 73.0
        resin_age = 5.0

    return {
        "ambient_temp": ambient_temp,
        "resin_temp": resin_temp,
        "resin_age": resin_age,
    }

# ==========================================================
#                Main Closed-Loop Function (3 runs overlay)
# ==========================================================
def main(num_runs=3, max_iterations=15, tolerance=0.005, simulate=True):
    num_runs = get_number_of_runs(default=3)
    print(f"Running {num_runs} independent runs")
    gs_logger = None
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    sheet_id = os.environ.get("SHEET_ID")
    worksheet = os.environ.get("WORKSHEET_NAME", "Dataset")

    if creds_path and sheet_id:
        try:
            gs_logger = GoogleSheetsLogger(creds_path, sheet_id, worksheet)
            print(f"[Sheets] Logging enabled -> tab '{worksheet}'")
        except Exception as e:
            print("[Sheets] Logging init failed:", e)
    else:
        print("[Sheets] Logging disabled")

    """Runs the closed-loop optimization multiple times."""

    all_histories = []  # list of CV lists, one per run

    for run in range(num_runs):
        # --- Reset environmental context ONCE per run ---
        base_context = get_initial_context_snapshot()
        print("Initial context snapshot:", base_context)

        # c_new will drift over iterations
        c_new = dict(base_context)

        print(f"\n========== RUN {run + 1}/{num_runs} ==========")
        best_cv = float("inf")
        best_df_batch = None

        # Load and retrain surrogate fresh each run
        df_hist = load_dataset()

        start_batch_id = int(pd.to_numeric(df_hist["batch_id"], errors="coerce").max()) + 1

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

        # --- Train on historical dataset ---
        X, y, df_batches = update_training_data(df_hist)
        cbo.train_surrogate(X, y, verbose=True)
        print(f"Loaded dataset with {len(df_batches)} batches.")

        r2 = r2_score(y, cbo.model.predict(X))
        print(f"Initial R2_train = {r2:.3f}")

        # --- Initialize CV history ---
        prev_cv = float(df_batches["cv"].iloc[-1])
        cv_history = [prev_cv]

        # Track decoded settings we've already tested in THIS RUN
        seen = set()


        # --- Optimization loop ---
        for i in range(1, max_iterations + 1):
            print(f"\n--- Iteration {i} ---")

            # Context drift
            c_new["ambient_temp"] -= 0.2
            c_new["resin_temp"] -= 0.1
            c_new["resin_age"] += 0.2
            print(f"Context snapshot: {c_new}")

            # Persistent BO step
            best_params, best_lcb, _ = cbo.compute_bayes_opt(c_new, verbose=True)

            # Decode to physical values for experiment
            best_params["layer_thickness_um"] = 50 if int(round(best_params["layer_thickness_um"])) == 0 else 100
            fit_vals = [-250, -150, -50, 0, 50, 150, 250]
            best_params["fit_adjustment"] = fit_vals[int(best_params["fit_adjustment"])]

            # ---------- NO-REPEAT (decoded) SETTINGS PATCH ----------
            # Key uses decoded discrete knobs + a binned orientation so near-duplicates count as repeats
            key = (
                int(best_params["layer_thickness_um"]),
                int(best_params["fit_adjustment"]),
                int(round(best_params["z_rotation_deg"] / 10.0) * 10)  # bin angle to nearest 10 degrees
            )

            if key in seen:
                print("Repeated decoded setting detected -> forcing exploration")

                # 1) Jitter orientation to escape repeats
                best_params["z_rotation_deg"] = float(np.clip(
                    best_params["z_rotation_deg"] + np.random.uniform(-15, 15),
                    0, 90
                ))

                # 2) Sometimes flip layer thickness
                if np.random.rand() < 0.50:
                    best_params["layer_thickness_um"] = 50 if best_params["layer_thickness_um"] == 100 else 100

                # 3) Sometimes jump to a different fit setting
                if np.random.rand() < 0.50:
                    best_params["fit_adjustment"] = int(np.random.choice([-250, -150, -50, 0, 50, 150, 250]))

                # recompute key after forcing exploration
                key = (
                    int(best_params["layer_thickness_um"]),
                    int(best_params["fit_adjustment"]),
                    int(round(best_params["z_rotation_deg"] / 10.0) * 10)
                )

            seen.add(key)
            # ---------- END NO-REPEAT PATCH ----------

            print("Suggested parameters:", best_params)
            print(f"Best LCB (surrogate score): {best_lcb:.6f}")

            # Run experiment and append batch
            clean_batch_id = start_batch_id + (i - 1)
            df_batch = run_experiment(best_params, c_new, batch_id=clean_batch_id, simulate=simulate)
        
            df_hist = pd.concat([df_hist, df_batch], ignore_index=True)

            # Update training data + retrain surrogate
            X, y, df_batches = update_training_data(df_hist)

            cbo.train_surrogate(X, y, verbose=False)

            r2 = r2_score(y, cbo.model.predict(X))
            print(f"Iter {i}: R2_train = {r2:.3f}")

            new_cv = float(df_batches["cv"].iloc[-1])

            # track best batch (so we can log ONLY the best one at the end of the run)
            if new_cv < best_cv:
                best_cv = new_cv
                best_df_batch = df_batch.copy()

            delta_cv = new_cv - prev_cv
            cv_history.append(new_cv)


            print(
                f"Previous CV: {prev_cv:.4f}, "
                f"New CV: {new_cv:.4f}, "
                f"delta_cv: {delta_cv:.4f}"
            )

            prev_cv = new_cv

        # Append ONE ROW per run: best parameters + mean flow
        if gs_logger is not None and best_df_batch is not None:
            try:
                r0 = best_df_batch.iloc[0]

                mean_flow = float(best_df_batch["channel_flow_rate_ml_per_min"].mean())

                one_row = {
                    "batch_id": int(r0["batch_id"]),
                    "channel_id": 0,  # 0 means "summary row"
                    "layer_thickness_um": int(r0["layer_thickness_um"]),
                    "z_rotation_deg": float(r0["z_rotation_deg"]),
                    "fit_adjustment": float(r0["fit_adjustment"]),
                    "resin_age": float(r0["resin_age"]),
                    "resin_temp": float(r0["resin_temp"]),
                    "ambient_temp": float(r0["ambient_temp"]),
                    # store mean as the single representative flow value
                    "channel_flow_rate_ml_per_min": mean_flow,
                }

                gs_logger.append_dataframe(pd.DataFrame([one_row]))
                print(f"[Sheets] Appended 1 BEST row for run {run + 1} (best CV = {best_cv:.4f})")
            except Exception as e:
                print("[Sheets] Append failed:", e)

        # store ONE history per run
        all_histories.append(cv_history)

    # --- Visualization ---
    visualize_model_convergence(all_histories)

    visualize_control_chart(df_batches)

if __name__ == "__main__":
    main()
