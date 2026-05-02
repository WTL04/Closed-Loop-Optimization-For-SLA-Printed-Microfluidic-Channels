import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cadquery as cq

from contextual_opt.src.pipeline.config import (
    NUM_CHANNELS,
    CHANNEL_LENGTH_BOUNDS,
    CHANNEL_WIDTH_BOUNDS,
    CHANNEL_HEIGHT_BOUNDS,
    MIN_CHANNEL_SPACING,
    BASE_WIDTH,
    BASE_LENGTH,
    BASE_THICKNESS,
)
from contextual_opt.src.core.ax_cbo import ContextualBayesOptAx
from ax.core import (
    SearchSpace,
    RangeParameter,
    ChoiceParameter,
    ParameterType,
)
from ax.core.parameter_constraint import ParameterConstraint
from contextual_opt.src.api.sheets_api import pullData, get_latest_col_value, append_row


def build_search_space(num_channels: int = NUM_CHANNELS):
    """
    Build search space with independent parameters for each channel.

    Parameters:
        num_channels: Number of independent channels (default: 4)

    Returns:
        SearchSpace with:
        - 3 × num_channels channel dimension parameters (TUNABLE)
        - 1 layer thickness parameter (TUNABLE)
        - 3 context parameters: ambient_temp, resin_temp, resin_age (FIXED)
        - 3 × num_channels post-print warp delta parameters (FIXED)
    """
    parameters = []

    # Shared printing parameter
    parameters.append(
        ChoiceParameter(
            name="layer_thickness_um",
            parameter_type=ParameterType.INT,
            values=[50, 100],
            is_ordered=True,
            sort_values=True,
        )
    )

    # Per-channel geometric parameters
    for i in range(1, num_channels + 1):
        parameters.extend(
            [
                RangeParameter(
                    name=f"channel_{i}_length",
                    parameter_type=ParameterType.FLOAT,
                    lower=CHANNEL_LENGTH_BOUNDS[0],
                    upper=CHANNEL_LENGTH_BOUNDS[1],
                ),
                RangeParameter(
                    name=f"channel_{i}_width",
                    parameter_type=ParameterType.FLOAT,
                    lower=CHANNEL_WIDTH_BOUNDS[0],
                    upper=CHANNEL_WIDTH_BOUNDS[1],
                ),
                RangeParameter(
                    name=f"channel_{i}_height",
                    parameter_type=ParameterType.FLOAT,
                    lower=CHANNEL_HEIGHT_BOUNDS[0],
                    upper=CHANNEL_HEIGHT_BOUNDS[1],
                ),
            ]
        )

    # Context parameters (fixed during optimization, vary between experiments)
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

    # post-print warp deltas (measured - intended) from the previous print.
    # fixed at suggestion time via ObservationFeatures, never optimized over.
    # stored in search space so add_historical() can index them by name.
    for i in range(1, num_channels + 1):
        parameters.extend(
            [
                RangeParameter(
                    name=f"channel_{i}_post_print_length_delta",
                    parameter_type=ParameterType.FLOAT,
                    lower=-0.3,  # ~2% of 15mm nominal
                    upper=0.3,
                ),
                RangeParameter(
                    name=f"channel_{i}_post_print_width_delta",
                    parameter_type=ParameterType.FLOAT,
                    lower=-0.05,  # ~15% of 0.3mm nominal — microchannels have worse relative error
                    upper=0.05,
                ),
                RangeParameter(
                    name=f"channel_{i}_post_print_height_delta",
                    parameter_type=ParameterType.FLOAT,
                    lower=-0.05,  # same reasoning as width
                    upper=0.05,
                ),
            ]
        )

    # Constraint: sum of all channel widths + minimum spacing must fit in base
    # sum(channel_i_width) + (num_channels - 1) * MIN_SPACING <= BASE_WIDTH
    # Rearranged: sum(channel_i_width) <= BASE_WIDTH - (num_channels - 1) * MIN_SPACING
    max_total_width = BASE_WIDTH - (num_channels - 1) * MIN_CHANNEL_SPACING

    # Build constraint string: "channel_1_width + channel_2_width + ... <= max_total_width"
    width_terms = " + ".join([f"channel_{i}_width" for i in range(1, num_channels + 1)])
    constraint_str = f"{width_terms} <= {max_total_width}"

    width_constraint = ParameterConstraint(inequality=constraint_str)

    search_space = SearchSpace(
        parameters=parameters,
        parameter_constraints=[width_constraint],
    )

    return search_space


def extract_channel_params(
    params: dict, num_channels: int = NUM_CHANNELS
) -> list[dict]:
    """
    Extract individual channel parameters from flat parameter dict.

    Args:
        params: Flat dict with keys like 'channel_1_length', 'channel_2_width', etc.
        num_channels: Number of channels

    Returns:
        List of dicts, one per channel: [{'length': ..., 'width': ..., 'height': ...}, ...]
    """
    channels = []
    for i in range(1, num_channels + 1):
        channels.append(
            {
                "length": params[f"channel_{i}_length"],
                "width": params[f"channel_{i}_width"],
                "height": params[f"channel_{i}_height"],
            }
        )
    return channels


def compute_flow_rate_cv(flow_rates: list[float]) -> float:
    """
    Compute coefficient of variation (CV) for flow rate uniformity.

    CV = std / mean (lower is better, indicates more uniform flow)

    Args:
        flow_rates: List of flow rates for each channel

    Returns:
        Coefficient of variation (0 = perfectly uniform)
    """
    arr = np.array(flow_rates)
    mean = np.mean(arr)
    if mean == 0:
        return float("inf")
    return float(np.std(arr) / mean)


def load_dataset(
    is_testing: bool, sheet_name: str = "Experiment Random Deltas", verbose=True
):
    """
    Returns DataFrame from Google Spreadsheet or a chosen fake dataset.

    Args:
        is_testing: bool
            Uses fake dataset when True, Google Spreadsheet when False
        sheet_name: str
            Name of the Google Sheets tab to load from (default: "Experiment Random Deltas")
    """
    if is_testing:
        print(
            "\n1) dataset_30_batches.csv \n2) dataset_5_batches.csv \n3) dataset_10_batches.csv \n4) dataset_15_batches.csv \n5) experiment_realistic_deltas.csv \n6) experiment_random_deltas.csv"
        )
        choice = input("\nPlease choose one of the six: ")

        if choice == "1":
            path = "contextual_opt/datasets/dataset_30_batches.csv"
        elif choice == "2":
            path = "contextual_opt/datasets/dataset_5_batches.csv"
        elif choice == "3":
            path = "contextual_opt/datasets/dataset_10_batches.csv"
        elif choice == "4":
            path = "contextual_opt/datasets/dataset_15_batches.csv"
        elif choice == "5":
            path = "contextual_opt/datasets/experiment_realistic_deltas.csv"
        elif choice == "6":
            path = "contextual_opt/datasets/experiment_random_deltas.csv"
        else:
            raise ValueError("Invalid fake dataset option")

        if verbose:
            print(f"Loading fake dataset: {path}")
        return pd.read_csv(path)

    # pull from google sheets api
    return pullData(sheet_name=sheet_name, verbose=verbose)


def fake_objective(params: dict, context: dict, noise_std: float = 1.0) -> float:
    """
    Fake objective for testing only.
    """
    return float(np.random.normal(1e-6, noise_std))


def get_context_snapshot(prev_warp: dict | None = None) -> dict:
    """
    Get context snapshot either manually or use fixed testing values.
    Set warp deltas from previous print as context.
    """
    print(
        "\n1) Manually input context snapshot \n2) Use fixed testing context snapshot "
    )
    choice = input("\nPlease choose one of the two: ")

    if choice == "1":
        base = {
            "ambient_temp": float(input("ambient_temp (°F): ")),
            "resin_temp": float(input("resin_temp (°F): ")),
            "resin_age": float(input("resin_age (estimated hours since opened): ")),
        }
    elif choice == "2":
        base = {
            "ambient_temp": 80.0,
            "resin_temp": 80.0,
            "resin_age": 15.0,
        }
    else:
        raise ValueError("Invalid context option")

    # warp deltas from the previous print, zeroed out on first run
    if prev_warp is None:
        prev_warp = {
            f"channel_{i}_post_print_{dim}_delta": 0.0
            for i in range(1, NUM_CHANNELS + 1)
            for dim in ("length", "width", "height")
        }

    return {**base, **prev_warp}


def load_data_source(
    sheet_name: str = "Experiment Random Deltas", is_testing: bool = False
):
    """
    Load data from Google Sheets or fake testing data.
    """
    if is_testing:
        return False, load_dataset(is_testing=True, sheet_name=sheet_name, verbose=True)
    return True, load_dataset(is_testing=False, sheet_name=sheet_name, verbose=True)


def run_fake_trial(cbo, trial, context):
    """
    Run a fake trial for testing purposes.
    """
    suggested_params = trial.arms[0].parameters
    y = fake_objective(suggested_params, context)
    cbo.observe(trial=trial, metric_value=y)


# returns (bool, dict | None) to pass warp deltas forward to next trial
def run_real_trial(
    trial, context, num_channels: int = NUM_CHANNELS
) -> tuple[bool, dict | None, int]:
    """
    Run a real trial: save suggested params to JSON and append to Google Sheets.

    Returns:
        bool: True if trial completed successfully, False otherwise
        dict: warp deltas to pass forward to next trial
        int: batch_id for reference
    """
    suggested_params = trial.arms[0].parameters

    batch_raw = get_latest_col_value(column_name="batch_id", sheet_name="Ax")
    batch_id = int(batch_raw) if batch_raw is not None else 1
    batch_id += 1

    save_params_to_json(suggested_params, batch_id=batch_id)

    # append batch to channel dimensions to sheets
    append_row(batch_id, suggested_params, context, sheet_name="Ax")
    print(f"\nAppended batch {batch_id} to Google Sheets.")

    # set deltas values in sheets to default as 0.0
    warp_deltas = {}
    for i in range(1, num_channels + 1):
        for dim in ("length", "width", "height"):
            warp_deltas[f"channel_{i}_post_print_{dim}_delta"] = 0.0

    # set flow rate values in sheets to default as 0.0
    flow_rates = {}
    for i in range(1, num_channels + 1):
        flow_rates[f"channel_{i}_flow_rate_ml_per_min"] = 0.0
    flow_rates["flow_rate_cv"] = 0.0

    print("Run post_print.py to record measurements.")

    return True, warp_deltas, batch_id


def build_cad_model(params: dict, num_channels: int = NUM_CHANNELS):
    """
    Build CAD model with 4 independent channel rows.

    Channels are arranged as parallel rows along the Y-axis with minimum spacing.
    Each channel has its own length, width, and height.

    Args:
        params: Parameter dict from CBO with channel_i_length/width/height
        num_channels: Number of channels

    Returns:
        CadQuery Workplane object ready for export
    """
    channels = extract_channel_params(params, num_channels)

    # Create base plate
    base = cq.Workplane("XY").box(BASE_LENGTH, BASE_WIDTH, BASE_THICKNESS)

    # Calculate Y positions for each channel (distributed with minimum spacing)
    total_width = sum(ch["width"] for ch in channels)
    total_spacing = (num_channels - 1) * MIN_CHANNEL_SPACING

    # Start position (bottom of first channel from center)
    y_start = -(total_width + total_spacing) / 2

    # Cut each channel into the base
    current_y = y_start
    result = base

    for i, ch in enumerate(channels):
        # Channel center Y position
        channel_center_y = current_y + ch["width"] / 2

        # Cut into base from top surface
        channel_center_z = BASE_THICKNESS / 2  # center of base plate

        # Create channel cavity
        channel = (
            cq.Workplane("XY")
            .box(ch["length"], ch["width"], ch["height"])
            .translate((0, channel_center_y, channel_center_z))
        )

        result = result.cut(channel)

        # Move to next channel position
        current_y += ch["width"] + MIN_CHANNEL_SPACING

    return result


def export_cad_model(params: dict, filename: str = "../cad_models/channels_fluid.stl"):
    """
    Build and export CAD model from suggested params.
    """
    model = build_cad_model(params)
    cq.exporters.export(model, filename)
    print(f"CFD fluid-domain STL generated: {filename}")
    print(f"Path to Model: {filename}")


def visualize_convergence(cbo):
    """
    Plot optimization convergence trace.
    """
    trace = cbo.optimization_trace()
    plt.plot(trace["trial_index"], trace["best_so_far"])
    plt.title("CBO trial convergence")
    plt.xlabel("Trial")
    plt.ylabel("Best flow_rate_cv so far")
    plt.show()


def print_suggested_params(suggested_params: dict, num_channels: int = NUM_CHANNELS):
    """
    Print suggested parameters in a readable format.
    """
    channels = extract_channel_params(suggested_params, num_channels)

    print("\nSuggested parameters:")
    for i, ch in enumerate(channels, 1):
        print(
            f"  Channel {i}: L={ch['length']:.2f}, W={ch['width']:.2f}, H={ch['height']:.2f} mm"
        )
    print(f"  Layer thickness: {suggested_params['layer_thickness_um']} um")


def save_params_to_json(
    params: dict, batch_id: int = None, filename: str = "suggested_params.json"
):
    """
    Save suggested parameters to JSON file.
    """
    data = {"batch_id": batch_id, **params}
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {filename}")


def observe_previous_trial(cbo, sheet_name: str = "Ax"):
    """
    Check for previous trial with flow_rate_cv in Google Sheets and observe it.
    Returns the observed trial if found, None otherwise.
    """
    cv_raw = get_latest_col_value(column_name="flow_rate_cv", sheet_name=sheet_name)
    if cv_raw is not None and cv_raw != "":
        cv = float(cv_raw)
        trials = list(cbo.experiment.trials.values())
        if trials:
            trial = trials[-1]
            if trial.status.is_completed:
                print(f"\nPrevious trial already observed with flow_rate_cv: {cv:.6f}")
                return None
            cbo.observe(trial=trial, metric_value=cv)
            print(f"\nObserved previous trial with flow_rate_cv: {cv:.6f}")
            return trial
    return None


def main():
    try:
        prev_warp = None

        print("\n1) Use Google Sheets Data \n2) Use fake testing data")
        data_choice = input("\nPlease choose one of the two: ")

        sheet_name = "Experiment Random Deltas"
        if data_choice == "1":
            sheet_name = input(
                "\nEnter Google Sheet tab name (default: Experiment Random Deltas): "
            ).strip()
            if not sheet_name:
                sheet_name = "Experiment Random Deltas"

        context = get_context_snapshot(prev_warp=prev_warp)

        cbo = ContextualBayesOptAx(
            search_space=build_search_space(),
            metric_name="flow_rate_cv",
            minimize=True,
        )

        use_real_data, df = load_data_source(
            sheet_name=sheet_name, is_testing=(data_choice == "2")
        )
        cbo.add_historical(df)
        print("Loaded Dataset into CBO surrogate")

        if use_real_data:
            observe_previous_trial(cbo)

        result = cbo.suggest(isOnline=True, c_t=context)
        trial = result["trial"]
        suggested_params = trial.arms[0].parameters

        print_suggested_params(suggested_params)

        print("\n1) Export CAD Model \n2) Skip")
        cad_choice = input("\nPlease choose one of the two: ")

        if cad_choice == "1":
            export_cad_model(suggested_params)
        elif cad_choice != "2":
            raise ValueError("Invalid CAD option")

        if use_real_data:
            completed, prev_warp, batch_id = run_real_trial(trial, context)
            if not completed:
                return
            print(f"\nBatch {batch_id} saved. Run print, then post_print.py")
        else:
            run_fake_trial(cbo, trial, context)
            prev_warp = None

        visualize_convergence(cbo)

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
