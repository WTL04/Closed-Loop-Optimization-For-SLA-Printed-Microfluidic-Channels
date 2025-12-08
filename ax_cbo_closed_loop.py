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
                name="fit_adjustment_pct",
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


def load_dataset():
    return pd.read_csv("./datasets/dataset.csv")


def main():
    # fixed context snapshot
    ambient_temp = 75.0  # °F
    resin_temp = 73.0  # °F
    resin_age = 5.0  # days

    c_new = {
        "ambient_temp": ambient_temp,
        "resin_temp": resin_temp,
        "resin_age": resin_age,
    }

    # initialzing CBO search_space and experiment
    search_space = build_search_space()
    cbo = ContextualBayesOptAx(
        search_space=search_space, metric_name="flow_rate_per_min", minimize=True
    )

    # initilaize historical data
    df = load_dataset()
    cbo.add_historical(df)
    print("Loaded Dataset into CBO surrogate")

    # TODO: apply cbo.suggest(), cbo.observe, and cbo.best_point after finish implmenenting
    # to test out functions
    suggested = cbo.suggest(c_new)
    params = suggested["params"]
    print(params)

    # TODO: visualize cbo's improvment with plotting
    trace = cbo.optimization_trace()
    print(trace)

    plt.plot(trace["trial_index"], trace["best_so_far"])
    plt.xlabel("Trial")
    plt.ylabel("Best flow_rate_per_min so far")
    plt.show()


main()
