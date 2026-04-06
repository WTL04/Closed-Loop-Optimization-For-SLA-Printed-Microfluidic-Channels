"""ax/run_cbo.py"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    NUM_CHANNELS,
    CHANNEL_LENGTH_BOUNDS,
    CHANNEL_WIDTH_BOUNDS,
    CHANNEL_HEIGHT_BOUNDS,
    MIN_CHANNEL_SPACING,
    BASE_WIDTH,
)
from ax_cbo import ContextualBayesOptAx
from ax.core import (
    SearchSpace,
    RangeParameter,
    ChoiceParameter,
    ParameterType,
)
from ax.core.parameter_constraint import ParameterConstraint
from sheets_api import pullData, get_latest_col_value, append_row


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


def load_dataset(is_testing: bool, verbose=True):
    """
    Returns DataFrame from Google Spreadsheet or a chosen fake dataset.

    Args:
        is_testing: bool
            Uses fake dataset when True, Google Spreadsheet when False
    """
    if is_testing:
        choice = input(
            "Choose fake dataset: 1) dataset_30_batches.csv 2) dataset_5_batches.csv 3) dataset_10_batches.csv 4) dataset_15_batches.csv: "
        )

        if choice == "1":
            path = "../../datasets/dataset_30_batches.csv"
        elif choice == "2":
            path = "../../datasets/dataset_5_batches.csv"
        elif choice == "3":
            path = "../../datasets/dataset_10_batches.csv"
        elif choice == "4":
            path = "../../datasets/dataset_15_batches.csv"
        else:
            raise ValueError("Invalid fake dataset option")

        if verbose:
            print(f"Loading fake dataset: {path}")
        return pd.read_csv(path)

    # pull from google sheets api
    return pullData(sheet_name="Ax", verbose=verbose)


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
    choice = input(
        "1) Manually input context snapshot 2) Use fixed testing context snapshot: "
    )
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


def load_data_source():
    """
    Select data source: Google Sheets or fake testing data.
    """
    choice = input("1) Use Google Sheets Data 2) Use fake testing data: ")
    if choice == "1":
        return True, load_dataset(is_testing=False, verbose=True)
    if choice == "2":
        return False, load_dataset(is_testing=True, verbose=True)
    raise ValueError("Invalid data source option")


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
) -> tuple[bool, dict | None]:
    """
    Run a real trial: append suggested params to spreadsheet and wait for user to record results.

    Returns:
        bool: True if trial completed successfully, False otherwise
    """
    suggested_params = trial.arms[0].parameters

    # get latest metadata and values from spreadsheet
    batch_raw = get_latest_col_value(column_name="batch_id", sheet_name="Geo Test")
    batch_id = int(batch_raw) if batch_raw is not None else 1
    batch_id += 1

    append_row(batch_id, suggested_params, context, sheet_name="Geo Test")

    if input("Did the print finish? (y/n) ").lower() == "n":
        return False, None

    #  collect independant per-channel post-print measurements as deltas (measured - intended)
    print("\nEnter post-print measurements for each channel:")
    warp_deltas = {}
    for i in range(1, num_channels + 1):
        print(f"  Channel {i}:")
        for dim in ("length", "width", "height"):
            intended = suggested_params[f"channel_{i}_{dim}"]
            measured = float(
                input(f"    Measured {dim} (intended={intended:.3f} mm): ")
            )
            warp_deltas[f"channel_{i}_post_print_{dim}_delta"] = measured - intended

    if (
        input("Did you record the resulting CV into the spreadsheet? (y/n) ").lower()
        == "n"
    ):
        return False, None

    return True, warp_deltas


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


def save_params_to_json(params: dict, filename: str = "suggested_params.json"):
    """
    Save suggested parameters to JSON file.
    """
    with open(filename, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved to {filename}")


def main():
    try:
        # no warp history on first run
        prev_warp = None

        # Get context snapshot
        context = get_context_snapshot(prev_warp=prev_warp)

        # Initialize CBO with per-channel search space
        cbo = ContextualBayesOptAx(
            search_space=build_search_space(),
            metric_name="flow_rate_cv",  # Coefficient of variation - minimize for uniformity
            minimize=True,
        )

        # Load data source (real or fake)
        use_real_data, df = load_data_source()
        cbo.add_historical(df)
        print("Loaded Dataset into CBO surrogate")

        # Get suggestion from CBO
        result = cbo.suggest(isOnline=True, c_t=context)
        trial = result["trial"]
        suggested_params = trial.arms[0].parameters

        # Print per-channel parameters
        print_suggested_params(suggested_params)

        # Save to JSON
        save_params_to_json(suggested_params)

        if use_real_data:
            # unpack warp deltas to carry forward to next trial
            completed, prev_warp = run_real_trial(trial, context)
            if not completed:
                return
            cv = float(
                get_latest_col_value(column_name="flow_rate_cv", sheet_name="Geo Test")
            )
            cbo.observe(trial=trial, metric_value=cv)
        else:
            run_fake_trial(cbo, trial, context)
            prev_warp = None

        visualize_convergence(cbo)

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
