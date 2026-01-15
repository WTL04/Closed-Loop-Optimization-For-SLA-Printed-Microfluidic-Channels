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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from cbo import ContextualBayesOpt
from visualize import visualize_model_convergence


# ==========================================================
#          Load & merge datasets
# ==========================================================
def load_dataset():
    """Loads the new dataset (already complete)"""
    df = pd.read_csv("datasets/dataset.csv")

    # No support_mode column in new dataset
    return df



# ==========================================================
#          Simulate (or run) a new print batch
# ==========================================================
def run_experiment(params, context, channels_per_batch=10, simulate=True):
    """Simulates a batch print for now; later replaced with real printer integration"""
    # --- New realistic flow model using ONLY new dataset features ---

    mean_flow = 100.0  # baseline nominal flow rate

    # layer thickness (0=50 um, 1=100 um)
    lt_value = params["layer_thickness_um"]  # already decoded to 50 or 100
    lt_factor = 1.10 if lt_value == 50 else 0.90
    mean_flow *= lt_factor

    # orientation / z rotation effect
    ori = params["z_rotation_deg"]
    ori_factor = 1 - abs(ori - 45) / 200
    mean_flow *= ori_factor

    # fit adjustment (tight fit → smaller flow)
    fit = params["fit_adjustment_pct"]
    fit_factor = 1 - abs(fit) / 3000
    mean_flow *= fit_factor

    # resin temperature
    temp_factor = 1 + (context["resin_temp"] - 72) * 0.01
    mean_flow *= temp_factor

    # ambient temperature
    ambient_factor = 1 + (context["ambient_temp"] - 72) * 0.005
    mean_flow *= ambient_factor

    # resin aging effect
    age_factor = 1 - (context["resin_age"] * 0.002)
    mean_flow *= age_factor

    rows = []
    batch_id = f"RUN_{int(time.time())}"

    for i in range(channels_per_batch):
        if simulate:
            noise = np.random.normal(0, 0.05 * mean_flow)
            flow = max(mean_flow + noise, 1e-6)
        else:
            flow = float(input(f"Measured flow for channel {i + 1}: "))

        rows.append(
            {
                "batch_id": batch_id,
                "layer_thickness_um": lt_value,
                "z_rotation_deg": params.get("z_rotation_deg", 45),
                "fit_adjustment_pct": params.get("fit_adjustment_pct", 0.0),
                "resin_temp": context["resin_temp"],
                "ambient_temp": context["ambient_temp"],
                "resin_age": context["resin_age"],
                "channel_id": f"{batch_id}_CH{i + 1:02d}",
                "flow_rate_per_min": flow,
            }
        )
    return pd.DataFrame(rows)


# ==========================================================
#             Compute CV & prep training data
# ==========================================================
def update_training_data(df_all):
    """Computes CV per batch, returns updated training data"""
    summary = (
        df_all.groupby("batch_id")["flow_rate_per_min"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["cv"] = summary["std"] / summary["mean"]

    features = [
        "layer_thickness_um",
        "z_rotation_deg",
        "fit_adjustment_pct",
        "resin_temp",
        "ambient_temp",
        "resin_age",
    ]

    df_batches = (
        df_all.groupby("batch_id")
        .first()[features]
        .reset_index()
        .merge(summary[["batch_id", "cv"]], on="batch_id")
    )
    df_batches["layer_thickness_um"] = df_batches["layer_thickness_um"].replace({50: 0, 100: 1})

    # Encode fit adjustment as categorical index
    fit_map = {v: i for i, v in enumerate([-250, -150, -50, 0, 50, 150, 250])}
    df_batches["fit_adjustment_pct"] = df_batches["fit_adjustment_pct"].map(fit_map)

    X, y = df_batches[features], df_batches["cv"]
    return X, y, df_batches


# ==========================================================
#                Main Closed-Loop Function (3 runs overlay)
# ==========================================================
def main(num_runs=3, max_iterations=15, tolerance=0.005, simulate=True):
    """Runs the closed-loop optimization multiple times under fixed context"""

    # --- Fixed environmental context ---
    ambient_temp = 75.0  # °F
    resin_temp = 73.0  # °F
    resin_age = 5.0  # days

    c_new = {
        "ambient_temp": ambient_temp,
        "resin_temp": resin_temp,
        "resin_age": resin_age,
    }

    all_histories = []  # list of CV trajectories for each run

    for run in range(num_runs):
        print(f"\n========== RUN {run + 1}/{num_runs} ==========")

        # Load and retrain surrogate fresh each run
        df_hist = load_dataset()

        features = [
            "layer_thickness_um",
            "z_rotation_deg",
            "fit_adjustment_pct",
            "resin_temp",
            "ambient_temp",
            "resin_age",
        ]
        categorical = ["layer_thickness_um", "fit_adjustment_pct"]

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
        pipeline = Pipeline(
            [
                ("preprocess", preprocess),
                ("rf", RandomForestRegressor(random_state=42 + run)),
            ]
        )

        pbounds = {
            "layer_thickness_um": (0, 1),    # categorical 0/1
            "z_rotation_deg": (0, 90),
            "fit_adjustment_pct": (0, 6),    # categorical index from 0–6
            }


        cbo = ContextualBayesOpt(pipeline=pipeline, pbounds=pbounds)

        # Train on historical dataset
        X, y, df_batches = update_training_data(df_hist)
        cbo.train_surrogate(X, y, verbose=True)
        print(f"Loaded dataset with {len(df_batches)} batches.")

        prev_cv = df_batches["cv"].iloc[-1]
        cv_history = [prev_cv]
        # stable_count = 0

        # --- Optimization loop with early stopping---
        for i in range(1, max_iterations + 1):
            print(f"\n--- Iteration {i} ---")
            print(f"Context snapshot: {c_new}")

            best_params, _, _ = cbo.compute_bayes_opt(c_new, verbose=True)
            # convert categorical layer thickness to real MoonRay values
            best_params["layer_thickness_um"] = 50 if int(best_params["layer_thickness_um"]) == 0 else 100

            # Decode categorical fit adjustment
            fit_vals = [-250, -150, -50, 0, 50, 150, 250]
            best_params["fit_adjustment_pct"] = fit_vals[int(best_params["fit_adjustment_pct"])]

            print("Suggested parameters:", best_params)

            df_batch = run_experiment(best_params, c_new, simulate=simulate)
            df_hist = pd.concat([df_hist, df_batch], ignore_index=True)

            X, y, df_batches = update_training_data(df_hist)
            cbo.train_surrogate(X, y, verbose=False)

            new_cv = df_batches["cv"].iloc[-1]
            cv_change = prev_cv - new_cv
            cv_history.append(new_cv)

            print(
                f"Previous CV: {prev_cv:.4f}, New CV: {new_cv:.4f}, ΔCV: {cv_change:.4f}"
            )

            # if abs(cv_change) < tolerance:
            #     stable_count += 1
            #     print(f"No significant improvement ({stable_count}/3)")
            # else:
            #     stable_count = 0
            #
            # prev_cv = new_cv
            #
            # if stable_count >= 3:
            #     print("CV stabilized; stopping early.")
            #     break

        all_histories.append(cv_history)

    # --- Visualization ---
    visualize_model_convergence(all_histories)


if __name__ == "__main__":
    main()
