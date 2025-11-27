import time
import numpy as np
import pandas as pd

from ax.core import (
    SearchSpace,
    RangeParameter,
    ChoiceParameter,
    ParameterType,
)
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax_cbo import ContextualBayesOptAx


def build_search_space():
    """
    Returns parameter search space in the context of SLA printing with Rayware
    """
    search_space = SearchSpace(
        parameters=[
            ChoiceParameter(
                name="layer_thickenss_um",
                parameter_type=ParameterType.INT,
                values=[50, 100],
            ),
            ChoiceParameter(
                name="fit_adjustment_pct",
                parameter_type=ParameterType.INT,
                values=[-250, -150, -50, 0, 50, 150, 250],
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
        search_space=search_space, metric_name="cv", minimize=True
    )

    # initilaize historical data
    df = load_dataset()
    cbo.add_historical(df)

    # TODO: apply cbo.suggest(), cbo.observe, and cbo.best_point after finish implmenenting
    # to test out functions

    # TODO: visualize cbo's improvment with plotting


main()
