"""
bayes_run_cbo.py
----------
Contextual Bayesian Optimization to generate suggested_parameters.json
Output: suggested_parameters.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # CLAUDE ADDED: needed for visualize_convergence
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from cbo import ContextualBayesOpt
from sheets_api import (
    pullData,
    get_latest_col_value,
    append_row,
)  # CLAUDE ADDED: get_latest_col_value, append_row for run_real_trial
from config import (
    NUM_CHANNELS,
    CHANNEL_LENGTH_BOUNDS,
    CHANNEL_WIDTH_BOUNDS,
    CHANNEL_HEIGHT_BOUNDS,
)


# ----- Set Context -----
def get_context_snapshot():
    """
    Get context snapshot either manually or use fixed testing values.
    """
    choice = input(
        "1) Manually input context snapshot 2) Use fixed testing context snapshot: "
    )
    if choice == "1":
        return {
            "ambient_temp": float(input("ambient_temp (°F): ")),
            "resin_temp": float(input("resin_temp (°F): ")),
            "resin_age": float(input("resin_age (estimated hours since opened): ")),
        }
    if choice == "2":
        return {
            "ambient_temp": 80.0,
            "resin_temp": 80.0,
            "resin_age": 15.0,
        }
    raise ValueError("Invalid context option")


# ----- Data Helpers -----
def load_dataset(is_testing: bool, verbose=True):
    """
    Returns DataFrame from Google Spreadsheet or a chosen fake dataset.

    Args:
        is_testing: bool
            Uses fake dataset when True, Google Spreadsheet when False
    """
    if is_testing:
        choice = input(
            "Choose fake dataset: 1) dataset_5_batches.csv 2) dataset_10_batches.csv 3) dataset_15_batches.csv 4) dataset_30_batches.csv: "
        )

        if choice == "1":
            path = "../datasets/dataset_5_batches.csv"
        elif choice == "2":
            path = "../datasets/dataset_10_batches.csv"
        elif choice == "3":
            path = "../datasets/dataset_15_batches.csv"
        elif choice == "4":
            path = "../datasets/dataset_30_batches.csv"
        else:
            raise ValueError("Invalid fake dataset option")

        if verbose:
            print(f"Loading fake dataset: {path}")
        return pd.read_csv(path)

    # pull from google sheets api
    return pullData(sheet_name="Bayes Opt", verbose=verbose)


def load_data_source() -> tuple[bool, pd.DataFrame]:
    """Prompt user to pick real or fake data, return (use_real_data, df)."""
    choice = input("1) Use Google Sheets data  2) Use fake testing data: ").strip()
    if choice == "1":
        return (
            True,
            load_dataset(is_testing=False, verbose=True),
        )  # CLAUDE ADDED: flipped bool to match ax/run_cbo.py convention (True = real data)
    if choice == "2":
        return (
            False,
            load_dataset(is_testing=True, verbose=True),
        )  # CLAUDE ADDED: flipped bool to match ax/run_cbo.py convention (False = fake data)
    raise ValueError("Invalid data source option")


# ----- Training Data Preparation -----
def prepare_training_data(df: pd.DataFrame, num_channels: int = NUM_CHANNELS):
    """
    Prepare training data from wide-format CSV where each row is one batch.

    Expected CSV structure (one row per batch):
        batch_id, layer_thickness_um, ambient_temp, resin_temp, resin_age,
        channel_{i}_length, channel_{i}_width, channel_{i}_height,   (per channel)
        channel_{i}_flow_rate_ml_per_min,                             (per channel)
        flow_rate_cv                                                   (pre-computed)

    CV is read directly from the `flow_rate_cv` column when present.
    If missing, it is computed from the per-channel flow rate columns.

    Returns:
        X          : pd.DataFrame  – one row per batch, feature columns only
        y          : pd.Series     – CV target per batch
        df_batches : pd.DataFrame  – full batch-level df (features + cv)
    """
    df = df.copy()

    # 1. Resolve CV column
    if "flow_rate_cv" in df.columns:
        # Use pre-computed CV; drop rows where it is missing
        df["flow_rate_cv"] = pd.to_numeric(df["flow_rate_cv"], errors="coerce")
        df_valid = df.dropna(subset=["flow_rate_cv"]).copy()

        if df_valid.empty:
            raise ValueError(
                "No rows with a valid 'flow_rate_cv' found. "
                "Please ensure the column is populated."
            )
    else:
        # Fall back: compute CV from per-channel flow rate columns
        flow_cols = [
            c
            for c in df.columns
            if c.startswith("channel_") and c.endswith("_flow_rate_ml_per_min")
        ]
        if not flow_cols:
            raise ValueError(
                "Dataset has neither a 'flow_rate_cv' column nor any "
                "'channel_{i}_flow_rate_ml_per_min' columns."
            )

        for col in flow_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df_valid = df.dropna(subset=flow_cols, how="all").copy()

        if df_valid.empty:
            raise ValueError(
                "No completed flow measurements found in dataset. "
                "Please ensure flow rates are recorded."
            )

        flow_vals = df_valid[flow_cols]
        row_mean = flow_vals.mean(axis=1)
        row_std = flow_vals.std(axis=1, ddof=1)
        df_valid["flow_rate_cv"] = row_std / row_mean

    # 2. Build feature column list (only columns present in the data)
    geo_features = [
        f"channel_{i}_{dim}"
        for i in range(1, num_channels + 1)
        for dim in ("length", "width", "height")
    ]
    context_features = ["resin_temp", "ambient_temp", "resin_age"]
    desired_features = ["layer_thickness_um"] + geo_features + context_features
    present_features = [f for f in desired_features if f in df_valid.columns]

    # 3. Assemble df_batches
    df_batches = (
        df_valid[present_features + ["flow_rate_cv"]].copy().reset_index(drop=True)
    )

    # Encode layer_thickness_um: 50 µm → 0, 100 µm → 1
    df_batches["layer_thickness_um"] = pd.to_numeric(
        df_batches["layer_thickness_um"], errors="coerce"
    ).replace({50: 0, 100: 1})

    X = df_batches[present_features]
    y = df_batches["flow_rate_cv"]
    return X, y, df_batches


# ------ Surrogate Model Pipeline Builder ------
def build_pipeline(features: list[str], run_seed: int = 42) -> Pipeline:
    """
    Build sklearn preprocessing + RandomForest pipeline.

    layer_thickness_um is one-hot encoded; all others are scaled.
    """
    categorical = ["layer_thickness_um"]
    numerical = [f for f in features if f not in categorical]

    preprocess = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("num", StandardScaler(), numerical),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("rf", RandomForestRegressor(random_state=run_seed)),
        ]
    )


# ------ Parameter decode helpers (encoded → physical) ------
def decode_params(raw_params: dict) -> dict:
    """
    Convert encoded parameter values back to physical / interpretable values.

    Encoding conventions (must match cbo_closed_loop.py):
        layer_thickness_um : 0 → 50 µm, 1 → 100 µm
        All channel geometry and context values are used as-is (continuous).
    """
    decoded = dict(raw_params)

    # layer_thickness_um: round to nearest int, map 0→50, else→100
    lt_enc = int(round(float(decoded.get("layer_thickness_um", 0))))
    decoded["layer_thickness_um"] = 50 if lt_enc == 0 else 100

    return decoded


def extract_channel_params(
    params: dict, num_channels: int = NUM_CHANNELS
) -> list[dict]:
    """
    Extract per-channel geometry dicts from a flat parameter dict.

    Returns a list like:
        [{'length': ..., 'width': ..., 'height': ...}, ...]
    """
    return [
        {
            "length": params[f"channel_{i}_length"],
            "width": params[f"channel_{i}_width"],
            "height": params[f"channel_{i}_height"],
        }
        for i in range(1, num_channels + 1)
    ]


# ------ Output helpers ------
def print_suggested_params(params: dict, num_channels: int = NUM_CHANNELS) -> None:
    """Pretty-print suggested parameters."""
    channels = extract_channel_params(params, num_channels)
    print("\nSuggested parameters:")
    for i, ch in enumerate(channels, 1):
        print(
            f"  Channel {i}: "
            f"L={ch['length']:.3f} mm, "
            f"W={ch['width']:.3f} mm, "
            f"H={ch['height']:.3f} mm"
        )
    print(f"  Layer thickness: {params['layer_thickness_um']} µm")


def save_params_to_json(
    params: dict, filename: str = "suggested_parameters.json"
) -> None:
    with open(filename, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n[Output] Saved suggested parameters → {filename}")


# CLAUDE ADDED: run_fake_trial — mirrors ax/run_cbo.py fake trial for testing
def run_fake_trial(best_params: dict, context: dict) -> float:
    """Run a fake trial for testing purposes — returns a random CV."""
    return float(np.random.normal(1e-6, 1.0))


# CLAUDE ADDED: run_real_trial — mirrors ax/run_cbo.py real trial flow:
# appends suggested params to spreadsheet, waits for print + CV confirmation
def run_real_trial(best_params: dict, context: dict) -> bool:
    """
    Append suggested params to spreadsheet and wait for user to record results.

    Returns:
        bool: True if trial completed successfully, False otherwise
    """
    batch_raw = get_latest_col_value(column_name="batch_id", sheet_name="Geo Test")
    batch_id = int(batch_raw) if batch_raw is not None else 1
    batch_id += 1

    append_row(batch_id, best_params, context, sheet_name="Geo Test")

    if input("Did the print finish? (y/n) ").lower() == "n":
        return False
    if (
        input("Did you record the resulting CV into the spreadsheet? (y/n) ").lower()
        == "n"
    ):
        return False

    return True


def visualize_convergence(lcb_history: list[float], cv_history: list[float]) -> None:
    """
    Plot two convergence views side by side:
      Left:  best LCB surrogate score per iteration
      Right: actual measured CV per iteration
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: LCB trace accumulated during the loop
    axes[0].plot(range(1, len(lcb_history) + 1), lcb_history, marker="o")
    axes[0].set_title("Surrogate convergence (LCB)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Best LCB so far")

    # Right: actual measured CV per iteration
    axes[1].plot(range(1, len(cv_history) + 1), cv_history, marker="o")
    axes[1].set_title("Measured CV per iteration")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("flow_rate_cv")

    plt.tight_layout()
    plt.show()


# ----- Main -----
if __name__ == "__main__":
    # load context
    context = get_context_snapshot()
    print(f"[Context] {context}")

    # load data
    use_real_data, df = load_data_source()
    print(f"[Data] Loaded {len(df)} rows, {df['batch_id'].nunique()} batches")

    # prepare training data
    X, y, df_batches = prepare_training_data(df, num_channels=NUM_CHANNELS)
    print(f"[Training] {len(df_batches)} complete batches available for training")

    features = list(X.columns)

    # define tunable parameters
    geo_pbounds = {
        f"channel_{i}_{dim}": bounds
        for i in range(1, NUM_CHANNELS + 1)
        for dim, bounds in [
            ("length", CHANNEL_LENGTH_BOUNDS),
            ("width", CHANNEL_WIDTH_BOUNDS),
            ("height", CHANNEL_HEIGHT_BOUNDS),
        ]
        if f"channel_{i}_{dim}" in features  # only include columns present in data
    }

    pbounds = {
        "layer_thickness_um": (0, 1),  # encoded: 0=50µm, 1=100µm
        **geo_pbounds,
    }

    # start CBO pipeline
    pipeline = build_pipeline(features, run_seed=42)
    cbo = ContextualBayesOpt(pipeline=pipeline, pbounds=pbounds, lam=2.0)

    # start surrogate training
    print("[Surrogate] Training … (grid search, may take a moment)")
    cbo.train_surrogate(X, y, verbose=True)
    print("[Surrogate] Training complete.")

    # CLAUDE ADDED: multi-iteration loop — mirrors cbo_closed_loop.py pattern.
    # Each iteration: suggest → trial → reload data → retrain → repeat.
    # cbo.optimizer is stateful (persists across compute_bayes_opt calls) so
    # each call adds ONE new BO step on top of all previous observations,
    # rather than restarting from scratch each iteration.
    max_iterations = 15
    best_cv = float("inf")
    best_params_overall = None
    cv_history = []  # CLAUDE ADDED: track measured CV per iteration for convergence plot
    lcb_history = []

    for i in range(1, max_iterations + 1):
        print(f"\n--- Iteration {i}/{max_iterations} ---")

        # return best parameters defined by CBO
        print("[BO] Running optimisation …")
        best_params_enc, best_lcb, _ = cbo.compute_bayes_opt(
            c_t=context,
            init_points=10
            if i == 1
            else 0,  # CLAUDE ADDED: random exploration only on first iteration
            n_iter=1,  # CLAUDE ADDED: one BO step per iteration; accumulates across loop via persistent optimizer
            verbose=True,
        )
        print(f"[BO] Best LCB (surrogate score): {best_lcb:.6f}")
        # decode parameters from encoded → physical values
        best_params = decode_params(best_params_enc)

        # save suggested parameters to suggested_parameters.json
        print_suggested_params(best_params, num_channels=NUM_CHANNELS)
        save_params_to_json(best_params)

        # CLAUDE ADDED: branch on real vs fake trial — mirrors ax/run_cbo.py
        if use_real_data:
            completed = run_real_trial(best_params, context)
            if not completed:
                print("[Trial] Aborted.")
                break
            cv = float(
                get_latest_col_value(column_name="flow_rate_cv", sheet_name="Geo Test")
            )
            print(f"[Trial] Recorded CV: {cv:.6f}")
        else:
            cv = run_fake_trial(best_params, context)
            print(f"[Trial] Fake CV: {cv:.6f}")

        # CLAUDE ADDED: track best params across iterations — mirrors cbo_closed_loop.py best_cv tracking
        cv_history.append(cv)
        if cv < best_cv:
            best_cv = cv
            best_params_overall = dict(best_params)

        # CLAUDE ADDED: track LCB each iteration for convergence plot
        lcb_history.append(best_lcb)

        # CLAUDE ADDED: reload and retrain surrogate on updated data after each trial —
        # mirrors cbo_closed_loop.py retrain loop so the model improves each iteration
        if use_real_data:
            df = load_dataset(is_testing=False, verbose=False)
        # (fake data does not grow between iterations, so no reload needed for testing)

        try:
            X, y, df_batches = prepare_training_data(df, num_channels=NUM_CHANNELS)
            cbo.train_surrogate(X, y, verbose=False)
            print(f"[Surrogate] Retrained on {len(df_batches)} batches.")
        except ValueError as e:
            print(f"[Training] Skipping retrain: {e}")

    # CLAUDE ADDED: print best result found across all iterations
    print("\n========== Optimization Complete ==========")
    print(f"Best CV achieved: {best_cv:.6f}")
    print(f"Best parameters: {best_params_overall}")

    # CLAUDE ADDED: visualize convergence trace — mirrors ax/run_cbo.py visualize_convergence call
    visualize_convergence(lcb_history, cv_history)
