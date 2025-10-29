#!/usr/bin/env python3
"""
phase2_closed_loop.py
----------------------------------------
Phase 2: Online / Closed-Loop Optimization
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
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from cbo import ContextualBayesOpt


# ==========================================================
#          Load & merge datasets like main.py
# ==========================================================
def load_dataset():
    """Loads and merges both datasets into one clean DataFrame"""
    df_multi = pd.read_csv("datasets/multi_batch_channels_dataset.csv")
    df_context = pd.read_csv("datasets/sla_spc_flowrate_channels_13batches.csv")

    # Add contextual features
    df_multi["resin_temp"] = df_context["resin_temp"]
    df_multi["resin_age"] = df_context["resin_age"]
    df_multi["ambient_temp"] = df_context["ambient_temp"]

    # Drop columns that don't apply to rectangular channels
    df_multi.drop(columns=["channel_diameter_mm"], inplace=True)

    return df_multi


# ==========================================================
#          Simulate (or run) a new print batch
# ==========================================================
def run_experiment(params, context, channels_per_batch=10, simulate=True):
    """Simulates a batch print for now; later replaced with real printer integration"""
    base_radius = params.get("channel_width_mm", 2.0) / 2.0
    resin_factor = 1.0 if params.get("resin_type", "Resin_A") == "Resin_A" else 0.92
    mean_flow = (
        resin_factor
        * (base_radius**4)
        / max(params.get("channel_length_mm", 30.0), 1.0)
        * 400.0
    )

    lt = params.get("layer_thickness_um", 50)
    ori = params.get("orientation_deg", 45)
    lt_effect = 1.05 if lt <= 20 else (0.95 if lt >= 100 else 1.0)
    ori_factor = 1.05 if 30 < ori < 60 else 0.97
    mean_flow *= lt_effect * ori_factor

    rows = []
    batch_id = f"RUN_{int(time.time())}"

    for i in range(channels_per_batch):
        if simulate:
            noise = np.random.normal(0, 0.1 * mean_flow)
            flow = max(mean_flow + noise, 1e-6)
        else:
            flow = float(input(f"Measured flow for channel {i + 1}: "))

        rows.append(
            {
                "batch_id": batch_id,
                "resin_type": params.get("resin_type", "Resin_A"),
                "layer_thickness_um": params.get("layer_thickness_um", 50),
                "orientation_deg": params.get("orientation_deg", 45),
                "support_mode": params.get("support_mode", "auto"),
                "fit_adjustment_pct": params.get("fit_adjustment_pct", 0.0),
                "channel_length_mm": params.get("channel_length_mm", 40.0),
                "channel_width_mm": params.get("channel_width_mm", 2.0),
                "resin_temp": context["resin_temp"],
                "ambient_temp": context["ambient_temp"],
                "resin_age": context["resin_age"],
                "channel_id": f"{batch_id}_CH{i + 1:02d}",
                "measured_flow_mL_per_min": flow,
            }
        )
    return pd.DataFrame(rows)


# ==========================================================
#             Compute CV & prep training data
# ==========================================================
def retrain_surrogate(df_all):
    """Computes CV per batch, returns updated training data"""
    summary = (
        df_all.groupby("batch_id")["measured_flow_mL_per_min"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["cv"] = summary["std"] / summary["mean"]

    features = [
        "resin_type",
        "layer_thickness_um",
        "orientation_deg",
        "support_mode",
        "fit_adjustment_pct",
        "channel_length_mm",
        "channel_width_mm",
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

    X, y = df_batches[features], df_batches["cv"]
    return X, y, df_batches


# ==========================================================
#                Main Closed-Loop Function
# ==========================================================
def main(max_iterations=15, tolerance=0.005, simulate=True):
    """Runs the closed-loop optimization until CV stops improving"""

    # Load historical dataset (combined version)
    df_hist = load_dataset()

    features = [
        "resin_type",
        "layer_thickness_um",
        "orientation_deg",
        "support_mode",
        "fit_adjustment_pct",
        "channel_length_mm",
        "channel_width_mm",
        "resin_temp",
        "ambient_temp",
        "resin_age",
    ]
    categorical = ["resin_type", "support_mode"]
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
        [("preprocess", preprocess), ("rf", RandomForestRegressor(random_state=42))]
    )

    pbounds = {
        "layer_thickness_um": (20, 100),
        "orientation_deg": (0, 90),
        "fit_adjustment_pct": (-2.0, 2.0),
        "channel_length_mm": (20, 60),
        "channel_width_mm": (1.0, 4.0),
    }

    cbo = ContextualBayesOpt(pipeline=pipeline, pbounds=pbounds)

    # Train on historical dataset
    X, y, df_batches = retrain_surrogate(df_hist)
    cbo.train_surrogate(X, y, verbose=True)
    print(f"Loaded dataset with {len(df_batches)} batches.")

    prev_cv = df_batches["cv"].iloc[-1]
    stable_count = 0
    cv_history = [prev_cv]  # store for plotting later

    for i in range(1, max_iterations + 1):
        print(f"\n========== Iteration {i} ==========")

        # --- Simulate cooling and resin aging ---
        if i == 1:
            ambient_temp = 75.0  # starting ambient temp (°F)
            resin_temp = 73.0  # starting resin temp (°F)
            resin_age = 5.0  # starting resin age (days)
        else:
            # simulate gradual cooling and aging each iteration
            ambient_temp -= 0.3  # ambient temperature drops slightly
            resin_temp -= 0.25  # resin cools with environment
            resin_age += np.random.uniform(0.5, 1.2)  # resin gets older

        c_new = {
            "ambient_temp": ambient_temp,
            "resin_temp": resin_temp,
            "resin_age": resin_age,
        }

        print(f"Context snapshot: {c_new}")

        best_params, _, _ = cbo.compute_bayes_opt(c_new, verbose=True)
        print("Suggested parameters:", best_params)

        df_batch = run_experiment(best_params, c_new, simulate=simulate)
        df_hist = pd.concat([df_hist, df_batch], ignore_index=True)

        X, y, df_batches = retrain_surrogate(df_hist)
        cbo.train_surrogate(X, y, verbose=False)

        new_cv = df_batches["cv"].iloc[-1]
        cv_change = prev_cv - new_cv
        cv_history.append(new_cv)

        print(f"Previous CV: {prev_cv:.4f}, New CV: {new_cv:.4f}, ΔCV: {cv_change:.4f}")

        if abs(cv_change) < tolerance:
            stable_count += 1
            print(f"No significant improvement ({stable_count}/3)")
        else:
            stable_count = 0

        prev_cv = new_cv

        if stable_count >= 3:
            print("\n CV has stabilized. Stopping optimization loop.")
            break

    df_hist.to_csv("datasets/multi_batch_channels_with_env_closedloop.csv", index=False)
    print("\nClosed-loop optimization complete and dataset updated!")

    # === Visualization ===
    plt.figure(figsize=(7, 4))
    plt.plot(
        range(len(cv_history)), cv_history, marker="o", linestyle="-", color="teal"
    )
    plt.title("Flow Rate CV Improvement Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Coefficient of Variation (CV)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
