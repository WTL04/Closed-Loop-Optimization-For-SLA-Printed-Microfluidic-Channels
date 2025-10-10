import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns


def visualize_control_chart(df_batches):
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(df_batches)+1), df_batches["cv"], marker="o", linestyle="-", label="Batch CVs")
    plt.axhline(df_batches["cv"].mean(), color="green", linestyle="--", label=f"Mean CV = {df_batches['cv'].mean():.3f}")
    plt.axhline(df_batches["cv"].mean() + 3*df_batches["cv"].std(), color="red", linestyle="--", label="UCL (Mean + 3σ)")
    plt.axhline(df_batches["cv"].mean() - 3*df_batches["cv"].std(), color="red", linestyle="--", label="LCL (Mean - 3σ)")
    plt.title("Control Chart for Batch CVs")
    plt.xlabel("Batch index")
    plt.ylabel("CV (std/mean)")
    plt.legend()
    plt.show()

def visualize_CV_Drift_relationship(df_batches):
    for c in ["ambient_temp", "resin_temp", "resin_age"]:
        sns.regplot(
            x=df_batches[c],
            y=df_batches["cv"],
            scatter_kws={"s": 50, "alpha": 0.7},
            line_kws={"color": "red"},
        )
        plt.title(f"Flow Rate (CV) vs Drift Context ({c}) ")
        plt.xlabel(c)
        plt.ylabel("CV")
        plt.show()

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X = df_batches[features] # inputs
    y = df_batches["cv"] # targets

    categorical = ["resin_type", "support_mode"]
    numerical = [f for f in features if f not in categorical]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numerical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    user_input = input("Visualze Drift Relationship (1) or Control Chart (2)?: ")

    if user_input == 1:
        visualize_CV_Drift_relationship(df_batches)
    elif user_input == 2:
        visualize_control_chart(df_batches)

    c_t = input("Enter Current Context: ")

    

if __name__=='__main__':
    main()
