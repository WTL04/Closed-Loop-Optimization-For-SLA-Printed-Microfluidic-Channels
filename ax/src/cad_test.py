import cadquery as cq
from ax_cbo import ContextualBayesOptAx
from ax.core import SearchSpace, RangeParameter, ChoiceParameter, ParameterType
import pandas as pd


def build_search_space():
    search_space = SearchSpace(
        parameters=[
            ChoiceParameter(
                name="layer_thickness_um",
                parameter_type=ParameterType.INT,
                values=[50, 100],
                is_ordered=True,
                sort_values=True,
            ),
            RangeParameter(
                name="channel_length",
                parameter_type=ParameterType.FLOAT,
                lower=30.0,
                upper=90.0,
            ),
            RangeParameter(
                name="channel_width",
                parameter_type=ParameterType.FLOAT,
                lower=10.0,
                upper=30.0,
            ),
            RangeParameter(
                name="channel_height",
                parameter_type=ParameterType.FLOAT,
                lower=10.0,
                upper=30.0,
            ),
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


def run_cbo_for_cad():
    context = {
        "ambient_temp": 80.0,
        "resin_temp": 80.0,
        "resin_age": 15.0,
    }

    cbo = ContextualBayesOptAx(
        search_space=build_search_space(),
        metric_name="channel_flow_rate_ml_per_min",
        minimize=True,
    )

    # Skip historical data - dataset doesn't have channel_length/width/height columns
    # df = pd.read_csv(
    #     "/home/will/Downloads/coding/uni/ml-research/contextual_bayes_opt/datasets/dataset.csv"
    # )
    # cbo.add_historical(df)
    print("Running CBO (cold start)")

    result = cbo.suggest(isOnline=False, c_t=context)
    trial = result["trial"]
    suggested_params = trial.arms[0].parameters

    print(f"Suggested parameters: {suggested_params}")
    return suggested_params


def build_cad_model(params: dict):
    channel_length = params["channel_length"]
    channel_width = params["channel_width"]
    channel_height = params["channel_height"]
    layer_thickness = params["layer_thickness_um"]

    base = cq.Workplane("XY").box(100, 100, 2)

    channel = (
        cq.Workplane("XY")
        .box(channel_length, channel_width, channel_height)
        .translate((0, 0, channel_height / 2 + 1))
    )

    result = base.cut(channel)

    return result


if __name__ == "__main__":
    params = run_cbo_for_cad()
    print(f"\nCAD Parameters: {params}")

    model = build_cad_model(params)

    print("\nExporting to STL...")
    cq.exporters.export(model, "/tmp/channel_test.stl")
    print("Exported to /tmp/channel_test.stl")
    print()
    print("Parameters for CAD modeling:")
    print(f"  - channel_length: {params['channel_length']}")
    print(f"  - channel_width: {params['channel_width']}")
    print(f"  - channel_height: {params['channel_height']}")
    print(f"  - layer_thickness_um: {params['layer_thickness_um']}")
