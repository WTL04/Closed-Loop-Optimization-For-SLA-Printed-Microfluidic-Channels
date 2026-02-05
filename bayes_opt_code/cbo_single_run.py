import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from cbo import ContextualBayesOpt
from visualize import visualize_control_chart, visualize_cv_drift_relationship
from pathlib import Path

def load_dataset():
    """Loads clean dataset including all features."""
    # Path to project root (one level above bayes_opt/)
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "datasets" / "dataset.csv"

    return pd.read_csv(csv_path)

def load_features():
    """
    Returns a list of knobs + context pertaining to the dataset
    """
    knobs = [
        "layer_thickness_um",
        "z_rotation_deg",
        "fit_adjustment",
    ]  # tunable knobs
    context = ["resin_age", "resin_temp", "ambient_temp"]  # drift context

    features = knobs + context
    return features

def main():
    current_ambient_temp = float(input(
        "Enter current ambient temperature (°F) as an integer: "
    ))
    current_resin_temp = float(input("Enter current resin temperature (°F) as an integer: "))
    current_resin_age = float(input(
        "Enter current number of days since opening resin container as an integer: "
    ))

    context_snapshot = {
        "ambient_temp": current_ambient_temp,
        "resin_temp": current_resin_temp,
        "resin_age": current_resin_age,
    }

    df_multi = load_dataset()
    features = load_features()

    # Compute per-batch mean, std, CV
    batch_summary = df_multi.groupby("batch_id")["channel_flow_rate_ml_per_min"].agg(
        ["mean", "std"]
    )
    batch_summary["cv"] = batch_summary["std"] / batch_summary["mean"]

    # Merge batch-level CV back with batch parameters
    df_batches = df_multi.groupby("batch_id").first()[features].reset_index()
    df_batches = df_batches.merge(batch_summary["cv"].reset_index(), on="batch_id")

    # Use mean historical CV as baseline
    baseline_cv = df_batches["cv"].mean()

    # ---Prep for surrogate training----
    df_batches["layer_thickness_um"] = df_batches["layer_thickness_um"].map({50: 0, 100: 1})
    fit_map = {v: i for i, v in enumerate([-250, -150, -50, 0, 50, 150, 250])}
    df_batches["fit_adjustment"] = df_batches["fit_adjustment"].replace(fit_map)

    X = df_batches[features]  # inputs
    y = df_batches["cv"]  # targets

    categorical = ["layer_thickness_um", "fit_adjustment"]
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

    # split train test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # BO search space (knobs)
    pbounds = {
        "layer_thickness_um": (0, 1),  # 0 = 50, 1 = 100 limited to these two as per dr.ava's comment
        "z_rotation_deg": (0, 90),
        "fit_adjustment": (0, 6),    # categorical index
    }

    # ---- BayesOpt Initial Exporlation---
    model = ContextualBayesOpt(pipeline=pipeline, pbounds=pbounds)
    model.train_surrogate(X_train, y_train, verbose=True)
    model.evaluate_surrogate(X_test, y_test)
    model.compute_bayes_opt(context_snapshot, verbose=True)
    
    best_params, _, _ = model.compute_bayes_opt(context_snapshot, verbose=False)


    # decode layer thickness
    best_params["layer_thickness_um"] = 50 if int(round(best_params["layer_thickness_um"])) == 0 else 100

    # decode fit adjustment
    fit_vals = [-250, -150, -50, 0, 50, 150, 250]
    best_params["fit_adjustment"] = fit_vals[int(best_params["fit_adjustment"])]

    print("\nDecoded best parameters:", best_params)


if __name__ == "__main__":
    main()