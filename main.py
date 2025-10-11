import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns

from cbo import ContextualBayesOpt
from visualize import visualize_control_chart, visualize_CV_Drift_relationship

def main():

    # Load the multi-batch dataset
    df_multi = pd.read_csv("datasets/multi_batch_channels_dataset.csv")
    df_context = pd.read_csv("datasets/sla_spc_flowrate_channels_13batches.csv")

    # add resin temp, resin age, ambient temp into df_multi
    df_multi["resin_temp"] = df_context["resin_temp"]
    df_multi["resin_age"] = df_context["resin_age"]
    df_multi["ambient_temp"] = df_context["ambient_temp"]


    #----Preprocess data----
    # drop unnessesary columns
    df_multi.drop(columns="channel_diameter_mm")

    # Compute per-batch mean, std, CV
    batch_summary = df_multi.groupby("batch_id")["measured_flow_mL_per_min"].agg(["mean", "std"])
    batch_summary["cv"] = batch_summary["std"] / batch_summary["mean"]

    # Select features (fixed per batch)
    knobs = ["resin_type", "layer_thickness_um", "orientation_deg", "support_mode", "fit_adjustment_pct"] # tunable knobs
    context = ["resin_age", "resin_temp", "ambient_temp"]  # drift context
    print_output = ["channel_length_mm", "channel_width_mm"] # output data from 3d print

    features = knobs + context # combine knobs with context

    # Merge batch-level CV back with batch parameters
    df_batches = df_multi.groupby("batch_id").first()[features].reset_index()
    df_batches = df_batches.merge(batch_summary["cv"].reset_index(), on="batch_id")

    # Use mean historical CV as baseline
    baseline_cv = df_batches["cv"].mean()


    #---Prep for model training----
    X = df_batches[features] # inputs
    y = df_batches["cv"] # targets

    categorical = ["resin_type", "support_mode"]
    numerical = [f for f in features if f not in categorical]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numerical)
    ])

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("rf", RandomForestRegressor(random_state=42))
    ])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # user_input = input("Visualze Drift Relationship (1) or Control Chart (2)?: ")
    #
    # if user_input == 1:
    #     visualize_CV_Drift_relationship(df_batches)
    # elif user_input == 2:
    #     visualize_control_chart(df_batches)
    #

    # BO search space (knobs)
    pbounds = {
        "layer_thickness_um": (50, 100), # limited to 50 and 100 per dr. ava's comment
        "orientation_deg": (0, 90),
        "fit_adjustment_pct": (-2.0, 2.0),
        "resin_type": ("Resin_A", "Resin_B", "Resin_C"), # A = 1, B = 2, C = 3, when encoded
        "support_mode": ("auto", "manual") # auto = 0, manual = 1, when encoded
    }

    # current context snapshot BEFORE a print (experimental)
    c_t = {
        "ambient_temp": 72,    # °F, lab measurement
        "resin_temp": 76,      # °F, sensor reading
        "resin_age": 12.0,        # days since resin was opened
    } 
    
    #---- BayesOpt Initial Exporlation---
    model = ContextualBayesOpt(pipeline=pipeline, pbounds=pbounds)
    model.train_surrogate(X_train, y_train, verbose=True)
    model.compute_bayes_opt(c_t, verbose=True)

    

if __name__=='__main__':
    main()
