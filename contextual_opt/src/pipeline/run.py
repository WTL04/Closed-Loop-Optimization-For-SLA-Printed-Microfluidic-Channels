"""
Main orchestration for running CBO with 4 channels.

Imports from decoupled modules:
- computation: compute_dimensional_error
- data_loader: load_data_source, extract_channel_data
- delta_loaders: generate_random_deltas, generate_realistic_deltas
- trial_runs: run_cfd_simulation
"""

import numpy as np
import json

from ax.core import (
    SearchSpace,
    RangeParameter,
    ChoiceParameter,
    ParameterType,
)

from contextual_opt.src.core.ax_cbo import ContextualBayesOptAx
from contextual_opt.src.pipeline.config import (
    CHANNEL_LENGTH_BOUNDS,
    CHANNEL_WIDTH_BOUNDS,
    CHANNEL_HEIGHT_BOUNDS,
    NOMINAL_DIMENSIONS,
)
from contextual_opt.src.api.sheets_api import get_latest_col_value, append_row
from contextual_opt.src.pipeline.metrics import compute_dimensional_error
from contextual_opt.src.pipeline.data_loader import (
    load_data_source,
    extract_channel_data,
)
from contextual_opt.src.pipeline.delta_loaders import (
    generate_random_deltas,
    generate_realistic_deltas,
)
from contextual_opt.src.pipeline.cfd_runs import run_cfd_simulation


def build_search_space(num_channels: int = 1):
    """Build search space for SINGLE channel optimization."""
    parameters = []

    parameters.append(
        ChoiceParameter(
            name="layer_thickness_um",
            parameter_type=ParameterType.INT,
            values=[50, 100],
            is_ordered=True,
            sort_values=True,
        )
    )

    parameters.extend(
        [
            RangeParameter(
                name="channel_length_mm",
                parameter_type=ParameterType.FLOAT,
                lower=CHANNEL_LENGTH_BOUNDS[0],
                upper=CHANNEL_LENGTH_BOUNDS[1],
            ),
            RangeParameter(
                name="channel_width_mm",
                parameter_type=ParameterType.FLOAT,
                lower=CHANNEL_WIDTH_BOUNDS[0],
                upper=CHANNEL_WIDTH_BOUNDS[1],
            ),
            RangeParameter(
                name="channel_height_mm",
                parameter_type=ParameterType.FLOAT,
                lower=CHANNEL_HEIGHT_BOUNDS[0],
                upper=CHANNEL_HEIGHT_BOUNDS[1],
            ),
        ]
    )

    parameters.extend(
        [
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

    # TODO: change range to 0-20 um
    parameters.extend(
        [
            RangeParameter(
                name="delta_length_um",
                parameter_type=ParameterType.FLOAT,
                lower=-0.3,
                upper=0.3,
            ),
            RangeParameter(
                name="delta_width_um",
                parameter_type=ParameterType.FLOAT,
                lower=-0.3,
                upper=0.3,
            ),
            RangeParameter(
                name="delta_height_um",
                parameter_type=ParameterType.FLOAT,
                lower=-0.3,
                upper=0.3,
            ),
        ]
    )

    return SearchSpace(parameters=parameters)


def print_suggested_params(suggested_params: dict):
    """Print suggested parameters in readable format."""
    print("\n=== Suggested Parameters ===")
    print(f"Length: {suggested_params.get('length', 'N/A'):.4f} mm")
    print(f"Width: {suggested_params.get('width', 'N/A'):.4f} mm")
    print(f"Height: {suggested_params.get('height', 'N/A'):.4f} mm")
    print(f"Layer Thickness: {suggested_params.get('layer_thickness_um', 'N/A')} µm")


def save_params_to_json(suggested_params: dict, batch_id: int):
    """Save suggested parameters to JSON file."""
    filename = "contextual_opt/src/data/suggested_params.json"
    data = {"batch_id": batch_id, **suggested_params}
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def get_context_snapshot(prev_warp=None):
    """Get context snapshot for CBO."""
    print(
        "\n1) Manually input context snapshot \n2) Use fixed testing context snapshot"
    )
    choice = input("\nPlease choose one of the two: ")

    if choice == "1":
        ambient = float(input("Ambient temperature (°C): "))
        resin = float(input("Resin temperature (°C): "))
        age = float(input("Resin age (hours): "))
        return {"ambient_temp": ambient, "resin_temp": resin, "resin_age": age}
    else:
        return {"ambient_temp": 80.0, "resin_temp": 80.0, "resin_age": 15.0}


def run_single_channel(
    cbo, context, sheet_name: str, channel_num: int, use_real_data: bool
):
    """
    Run CBO for a SINGLE channel, return result.

    Uses exact column names from dataset:
    batch_id, channel, layer_thickness_um, ambient_temp, resin_temp, resin_age,
    channel_length_mm, channel_width_mm, channel_height_mm,
    delta_length_um, delta_width_um, delta_height_um,
    dim_error, flow_rate
    """
    channel_context = {**context, "channel": channel_num}

    result = cbo.suggest(isOnline=True, c_t=channel_context)
    trial = result["trial"]
    suggested_params = trial.arms[0].parameters

    # Map to exact column names from dataset (with _mm suffix)
    channel_length_mm = suggested_params.get(
        "channel_length_mm", NOMINAL_DIMENSIONS["length"]
    )
    channel_width_mm = suggested_params.get(
        "channel_width_mm", NOMINAL_DIMENSIONS["width"]
    )
    channel_height_mm = suggested_params.get(
        "channel_height_mm", NOMINAL_DIMENSIONS["height"]
    )

    print(f"\n=== Channel {channel_num} ===")
    print(f"Length: {channel_length_mm:.4f} mm")
    print(f"Width: {channel_width_mm:.4f} mm")
    print(f"Height: {channel_height_mm:.4f} mm")
    print(f"Layer Thickness: {suggested_params.get('layer_thickness_um', 'N/A')} µm")

    if not use_real_data:
        # Fake trial
        flow_rate = np.random.uniform(0.1, 0.2)
        deltas = generate_random_deltas()
        channel_results = {
            "batch_id": channel_num,
            "channel": channel_num,
            "channel_length_mm": channel_length_mm,
            "channel_width_mm": channel_width_mm,
            "channel_height_mm": channel_height_mm,
            "delta_length_um": deltas["length"],
            "delta_width_um": deltas["width"],
            "delta_height_um": deltas["height"],
            "flow_rate": flow_rate,
        }
    else:
        # Real CFD
        deltas = (
            generate_realistic_deltas()
            if "Realistic" in sheet_name
            else generate_random_deltas()
        )
        flow_rate_m3s = run_cfd_simulation(
            length_delta=deltas["length"],
            width_delta=deltas["width"],
            height_delta=deltas["height"],
        )
        flow_rate_ml = flow_rate_m3s * 1e6 * 60

        channel_results = {
            "batch_id": channel_num,
            "channel": channel_num,
            "channel_length_mm": channel_length_mm,
            "channel_width_mm": channel_width_mm,
            "channel_height_mm": channel_height_mm,
            "delta_length_um": deltas["length"],
            "delta_width_um": deltas["width"],
            "delta_height_um": deltas["height"],
            "flow_rate": flow_rate_ml,
        }

        print(f"  Flow rate: {flow_rate_ml:.4f} mL/min")

    # Compute dimensional error (uses channel_* keys)
    dim_error = compute_dimensional_error(channel_results)
    channel_results["dim_error"] = dim_error

    # Observe result for CBO
    cbo.observe(trial=trial, metric_value=dim_error)

    return channel_results


def append_single_to_sheets(channel_results: dict, context: dict, sheet_name: str):
    """Append one row to Google Sheets. Use exact column names from dataset."""
    batch_raw = get_latest_col_value(column_name="batch_id", sheet_name=sheet_name)
    batch_id = int(batch_raw) + 1 if batch_raw is not None else 1

    row_data = {
        "batch_id": batch_id,
        "channel": channel_results.get("channel", 1),
        "layer_thickness_um": 100,
        "ambient_temp": context.get("ambient_temp", 80.0),
        "resin_temp": context.get("resin_temp", 80.0),
        "resin_age": context.get("resin_age", 15.0),
        "channel_length_mm": channel_results["channel_length_mm"],
        "channel_width_mm": channel_results["channel_width_mm"],
        "channel_height_mm": channel_results["channel_height_mm"],
        "delta_length_um": channel_results["delta_length_um"],
        "delta_width_um": channel_results["delta_width_um"],
        "delta_height_um": channel_results["delta_height_um"],
        "flow_rate": channel_results["flow_rate"],
    }
    append_row(batch_id, row_data, context, sheet_name=sheet_name)
    print(f"Appended batch {batch_id}, channel {row_data['channel']} to '{sheet_name}'")


def main():
    """Main entry point - runs ONE channel at a time."""
    try:
        print("\n1) Use Google Sheets Data \n2) Use fake testing data")
        data_choice = input("\nPlease choose one of the two: ")

        sheet_name = "Reformated - Experiment Realistic Deltas"
        if data_choice == "1":
            sheet_name = input("\nEnter Google Sheet tab name: ").strip()
            if not sheet_name:
                sheet_name = "Reformated - Experiment Realistic Deltas"

        # Ask which channel to run (1-4)
        print(f"\nWhich channel to optimize? (1-4)")
        channel_num = int(input("Channel: "))

        # define context
        context = get_context_snapshot()

        # initialize CBO
        cbo = ContextualBayesOptAx(
            search_space=build_search_space(),
            metric_name="dim_error",
            minimize=True,
        )

        # Load data and filter for this channel
        use_real_data, df_full = load_data_source(
            sheet_name=sheet_name, is_testing=(data_choice == "2")
        )
        df_channel = extract_channel_data(df_full, channel_num)
        cbo.add_historical(df_channel)
        print(f"Loaded {len(df_channel)} rows for channel {channel_num}")

        # Run CBO for ONE channel
        result = run_single_channel(
            cbo, context, sheet_name, channel_num, use_real_data=(data_choice == "1")
        )

        # Append result to sheets
        append_single_to_sheets(result, context, sheet_name)

        print(f"\nCompleted channel {channel_num}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
