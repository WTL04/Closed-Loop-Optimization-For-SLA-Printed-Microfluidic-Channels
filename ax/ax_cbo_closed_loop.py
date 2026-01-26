import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ax.core import (
    SearchSpace,
    RangeParameter,
    ChoiceParameter,
    ParameterType,
)
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax_cbo import ContextualBayesOptAx

from sheets_api import pullData, get_latest_col_value, append_row

# set default values to global vars
BATCH_ID = 1
NUM_CHANNELS = 13


def build_search_space():
    """
    Returns parameter search space in the context of SLA printing with Rayware
    """
    search_space = SearchSpace(
        parameters=[
            # -----------------
            # Decision variables
            # -----------------
            ChoiceParameter(
                name="layer_thickness_um",
                parameter_type=ParameterType.INT,
                values=[50, 100],
                is_ordered=True,
                sort_values=True,
            ),
            ChoiceParameter(
                name="fit_adjustment",
                parameter_type=ParameterType.INT,
                values=[-250, -150, -50, 0, 50, 150, 250],
                is_ordered=True,
                sort_values=True,
            ),
            RangeParameter(
                name="z_rotation_deg",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=90.0,
            ),
            # --------
            # Context
            # --------
            RangeParameter(
                name="ambient_temp",
                parameter_type=ParameterType.FLOAT,
                lower=60.0,
                upper=100.0,
            ),
            RangeParameter(
                name="resin_temp",
                parameter_type=ParameterType.FLOAT,
                lower=60.0,
                upper=100.0,
            ),
            RangeParameter(
                name="resin_age",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=72.0,
            ),
        ]
    )
    return search_space


def load_dataset(is_testing: bool, verbose=True):
    """
    Returns DataFrame from Google Spreadsheet or a chosen fake dataset.

    Args:
        is_testing: bool
            Uses fake dataset when True, Google Spreadsheet when False
    """
    if is_testing:
        choice = input(
            "Choose fake dataset: 1) dataset.csv 2) dataset_5_batches.csv 3) dataset_10_batches.csv 4) dataset_15_batches.csv: "
        )

        if choice == "1":
            path = "../datasets/dataset.csv"  # has 30 batches
        elif choice == "2":
            path = "../datasets/dataset_5_batches.csv"
        elif choice == "3":
            path = "../datasets/dataset_10_batches.csv"
        elif choice == "4":
            path = "../datasets/dataset_15_batches.csv"
        else:
            raise ValueError("Invalid fake dataset option")

        if verbose:
            print(f"Loading fake dataset: {path}")
        return pd.read_csv(path)

    return pullData(verbose=verbose)


def fake_objective(params: dict, context: dict, noise_std: float = 1.0) -> float:
    # -----------------------------
    # Fake objective (testing only)
    # -----------------------------
    return float(np.random.normal(1e-6, noise_std))


def get_context_snapshot():
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


def load_data_source():
    choice = input("1) Use Google Sheets Data 2) Use fake testing data: ")
    if choice == "1":
        return True, load_dataset(is_testing=False, verbose=True)
    if choice == "2":
        return False, load_dataset(is_testing=True, verbose=True)
    raise ValueError("Invalid data source option")


def run_fake_trial(cbo, trial, context):
    suggested_params = trial.arms[0].parameters
    fake_objective(suggested_params, context)


def run_real_trial(trial, context):
    suggested_params = trial.arms[0].parameters

    # get latest metadata and values from spreadsheet
    batch_raw = get_latest_col_value("batch_id")
    batch_id = int(batch_raw) if batch_raw is not None else 1
    batch_id += 1

    append_row(batch_id, NUM_CHANNELS, suggested_params, context)

    if input("Did the print finish? (y/n) ").lower() == "n":
        return False
    if (
        input("Did you record the resulting CV into the spreadsheet? (y/n) ").lower()
        == "n"
    ):
        return False

    return True


def visualize_convergence(cbo):
    trace = cbo.optimization_trace()
    plt.plot(trace["trial_index"], trace["best_so_far"])
    plt.title("CBO trial convergence")
    plt.xlabel("Trial")
    plt.ylabel("Best flow_rate_per_min so far")
    plt.show()


# ------------------------ testing ----------------------------
def run_single_experiment(dataset_path: str, context: dict, seed: int = 0):
    """
    Runs one experiment: load dataset, warm-start Ax with historical data,
    run one online suggestion + observation, and return the optimization trace.
    """
    np.random.seed(seed)

    df = pd.read_csv(dataset_path)

    cbo = ContextualBayesOptAx(
        search_space=build_search_space(),
        metric_name="channel_flow_rate_ml_per_min",
        minimize=True,
    )

    cbo.add_historical(df)

    # One online BO step (so you see behavior beyond just historical warm start)
    trial = cbo.suggest(isOnline=True, c_t=context)["trial"]
    suggested_params = trial.arms[0].parameters

    # Fake evaluation (testing): MUST observe or the trace won't include this trial
    # y = fake_objective(suggested_params, context, noise_std=1e-6)
    # cbo.observe(trial=trial, metric_value=y)

    return cbo.optimization_trace()


def run_all_4_experiments_and_plot(seed: int = 0, fixed_context: dict | None = None):
    """
    Runs 4 experiments (5, 10, 15, 30 batches) and plots them as 2x2 subplots.
    Returns (fig, axes).
    """
    if fixed_context is None:
        fixed_context = {"ambient_temp": 80.0, "resin_temp": 80.0, "resin_age": 15.0}

    experiments = [
        ("5 batches", "../datasets/dataset_5_batches.csv"),
        ("10 batches", "../datasets/dataset_10_batches.csv"),
        ("15 batches", "../datasets/dataset_15_batches.csv"),
        ("30 batches", "../datasets/dataset.csv"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, (title, path) in enumerate(experiments):
        trace = run_single_experiment(path, fixed_context, seed=seed + i)

        ax = axes[i]
        ax.plot(trace["trial_index"], trace["best_so_far"])
        ax.set_title(title)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best channel_flow_rate_ml_per_min so far")
        ax.grid(True, alpha=0.3)

    plt.suptitle("CBO trial convergence across batch sizes")
    plt.tight_layout()
    plt.show()

    return fig, axes


# ----------------------------------------------------------


def main():
    try:
        context = get_context_snapshot()
        cbo = ContextualBayesOptAx(
            search_space=build_search_space(),
            metric_name="channel_flow_rate_ml_per_min",
            minimize=True,
        )

        use_real_data, df = load_data_source()
        cbo.add_historical(df)
        print("Loaded Dataset into CBO surrogate")

        trial = cbo.suggest(isOnline=True, c_t=context)["trial"]

        if use_real_data:
            completed = run_real_trial(trial, context)
            if not completed:
                return
            cv = float(get_latest_col_value("channel_flow_rate_ml_per_min"))
            cbo.observe(trial=trial, metric_value=cv)
        else:
            run_fake_trial(cbo, trial, context)

        visualize_convergence(cbo)

    except ValueError as e:
        print(e)


main()
