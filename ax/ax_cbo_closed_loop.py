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

from sheets_api import pullData, get_latest_cv, append_row


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


def load_dataset(is_testing: bool):
    """
    Returns Dataframe from Google Spreadsheet or fake dataset

    Args:
        is_testing: bool
           Uses fake dataset when True, Google Spreadsheet when False
    """
    if is_testing:
        return pd.read_csv("../datasets/dataset.csv")

    return pullData(verbose=True)


def linear_context_change(c_new, strength=2):
    for value in c_new.values():
        if value == c_new["resin_age"]:
            value += 1  # resin age should increase every print
        value += strength

    return c_new


def main():
    # fixed context snapshot, debug
    ambient_temp = 80.0  # °F
    resin_temp = 80.0  # °F
    resin_age = 15.0  # hours

    c_new = {
        "ambient_temp": ambient_temp,
        "resin_temp": resin_temp,
        "resin_age": resin_age,
    }

    # initialzing CBO search_space and experiment
    search_space = build_search_space()
    cbo = ContextualBayesOptAx(
        search_space=search_space,
        metric_name="channel_flow_rate_ml_per_min",
        minimize=True,
    )

    # initilaize historical data
    df = load_dataset(is_testing=True)  # pulling fake data for now
    cbo.add_historical(df)
    print("Loaded Dataset into CBO surrogate")

    trial = cbo.suggest(isOnline=True, c_t=c_new)["trial"]  # online loop for testing
    suggested_params = trial.arms[0].parameters

    # auto add suggested parameters and context snapshot into spreadsheet
    append_row(suggested_params, c_new)

    user_check = input("Did you record the resulting CV into the spreadsheet? (y/n) ")
    if user_check == "n" or user_check == "N":
        return

    # GET recorded CV from Spreadsheet
    cv = get_latest_cv("channel_flow_rate_ml_per_min")

    # TODO: testing observe function
    cbo.observe(
        trial=trial, metric_value=cv
    )  # input resulting CV, mark trial as complete

    # visualize convergence
    trace = cbo.optimization_trace()
    print(trace)  # debug
    plt.plot(trace["trial_index"], trace["best_so_far"])
    plt.title("CBO trial convergence")
    plt.xlabel("Trial")
    plt.ylabel("Best flow_rate_per_min so far")
    plt.show()


main()
