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
CHANNEL_ID = 1


def build_search_space():
    """
    Returns parameter search space in the context of SLA printing with Rayware
    """
    search_space = SearchSpace(
        parameters=[
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
                parameter_type=ParameterType.FLOAT,  # could be changed to int
                lower=0.0,
                upper=90.0,
            ),
        ]
    )
    return search_space


def load_dataset(is_testing: bool, verbose=True):
    """
    Returns Dataframe from Google Spreadsheet or fake dataset

    Args:
        is_testing: bool
           Uses fake dataset when True, Google Spreadsheet when False
    """
    if is_testing:
        return pd.read_csv("../datasets/dataset.csv")

    return pullData(verbose=verbose)


def linear_context_change(c_new, strength=2):
    for value in c_new.values():
        if value == c_new["resin_age"]:
            value += 1  # resin age should increase every print
        value += strength

    return c_new


def main():
    # ---- context ----
    choice = input(
        "1) Manually input context snapshot 2) Use fixed testing context snapshot: "
    )
    if choice == "1":
        c_new = {
            "ambient_temp": float(input("ambient_temp (°F): ")),
            "resin_temp": float(input("resin_temp (°F): ")),
            "resin_age": float(input("resin_age (estimated hours since opened): ")),
        }
    elif choice == "2":
        c_new = {
            "ambient_temp": 80.0,
            "resin_temp": 80.0,
            "resin_age": 15.0,
        }
    else:
        print("not an option")
        return

    # ---- Ax init ----
    cbo = ContextualBayesOptAx(
        search_space=build_search_space(),
        metric_name="channel_flow_rate_ml_per_min",
        minimize=True,
    )

    # ---- data source ----
    choice = input("1) Use Google Sheets Data 2) Use fake testing data: ")
    if choice == "1":
        use_real_data = True
        df = load_dataset(is_testing=False, verbose=True)
    elif choice == "2":
        use_real_data = False
        df = load_dataset(is_testing=True, verbose=True)
    else:
        print("not an option")
        return

    cbo.add_historical(df)
    print("Loaded Dataset into CBO surrogate")

    # ---- suggest ----
    trial = cbo.suggest(isOnline=True, c_t=c_new)["trial"]

    if not use_real_data:
        print("Fake mode: suggestion only")
        return

    suggested_params = trial.arms[0].parameters

    # ---- metadata ----
    batch_raw = get_latest_col_value("batch_id")
    channel_raw = get_latest_col_value("channel_id")
    batch_id = int(batch_raw) if batch_raw is not None else 1
    channel_id = int(channel_raw) if channel_raw is not None else 1
    if channel_id >= 13:
        batch_id += 1
        channel_id = 1
    else:
        channel_id += 1

    append_row(batch_id, channel_id, suggested_params, c_new)

    if input("Did the print finish? (y/n) ").lower() == "n":
        return
    if (
        input("Did you record the resulting CV into the spreadsheet? (y/n) ").lower()
        == "n"
    ):
        return

    cv = float(get_latest_col_value("channel_flow_rate_ml_per_min"))
    cbo.observe(trial=trial, metric_value=cv)

    # ---- visualize convergence ----
    trace = cbo.optimization_trace()
    plt.plot(trace["trial_index"], trace["best_so_far"])
    plt.title("CBO trial convergence")
    plt.xlabel("Trial")
    plt.ylabel("Best flow_rate_per_min so far")
    plt.show()


main()
