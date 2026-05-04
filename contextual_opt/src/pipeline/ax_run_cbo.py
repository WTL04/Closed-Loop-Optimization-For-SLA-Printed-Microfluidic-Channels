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
    BASELINE_FLOW_RATE,
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

    # Channel position context parameter (which channel on the chip)
    parameters.append(
        ChoiceParameter(
            name="channel_position",
            parameter_type=ParameterType.INT,
            values=[1, 2, 3, 4],
            is_ordered=False,
            sort_values=True,
        )
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


def compute_dimensional_error(params: dict, num_channels: int = NUM_CHANNELS) -> float:
    """
    Compute mean squared error between intended and actual printed dimensions.

    Actual = Intended + Delta
    Error = mean(Delta_Length^2 + Delta_Width^2 + Delta_Height^2)

    Args:
        params: Dict with channel_i_length/width/height and channel_i_post_print_*_delta
        num_channels: Number of channels (default: NUM_CHANNELS from config)

    Returns:
        Mean squared error (lower is better)
    """
    squared_errors = []
    for i in range(1, num_channels + 1):
        try:
            length_delta = float(params.get(f"channel_{i}_post_print_length_delta", 0.0) or 0.0)
            width_delta = float(params.get(f"channel_{i}_post_print_width_delta", 0.0) or 0.0)
            height_delta = float(params.get(f"channel_{i}_post_print_height_delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            length_delta = 0.0
            width_delta = 0.0
            height_delta = 0.0

        squared_errors.extend(
            [
                length_delta ** 2,
                width_delta ** 2,
                height_delta ** 2,
            ]
        )

    return float(np.mean(squared_errors)) if squared_errors else 0.0


def calculate_functional_recovery(extracted_flow_rate: float) -> float:
    """
    Calculates how close the pre-distorted channel's flow rate
    is to the perfect nominal baseline.

    Args:
        extracted_flow_rate: Flow rate in m³/s from OpenFOAM

    Returns:
        Error percentage (0.0 = perfect recovery)
    """
    return abs(extracted_flow_rate - BASELINE_FLOW_RATE) / BASELINE_FLOW_RATE * 100.0


def melt_dataset_to_single_channel(
    df: pd.DataFrame, original_num_channels: int = 4
) -> pd.DataFrame:
    """
    Converts a batch-based dataset (4 channels per row) into a
    channel-based dataset (1 channel per row) for efficient 1D Bayesian Optimization.
    """
    shared_cols = [
        "batch_id",
        "layer_thickness_um",
        "ambient_temp",
        "resin_temp",
        "resin_age",
    ]
    melted_rows = []

    for i in range(1, original_num_channels + 1):
        temp_df = df[shared_cols].copy()

        temp_df["channel_position"] = i

        temp_df["channel_1_length"] = df[f"channel_{i}_length"]
        temp_df["channel_1_width"] = df[f"channel_{i}_width"]
        temp_df["channel_1_height"] = df[f"channel_{i}_height"]

        temp_df["channel_1_post_print_length_delta"] = df[
            f"channel_{i}_post_print_length_delta"
        ]
        temp_df["channel_1_post_print_width_delta"] = df[
            f"channel_{i}_post_print_width_delta"
        ]
        temp_df["channel_1_post_print_height_delta"] = df[
            f"channel_{i}_post_print_height_delta"
        ]

        temp_df["flow_rate"] = df[f"channel_{i}_flow_rate_ml_per_min"]

        melted_rows.append(temp_df)

    melted_df = pd.concat(melted_rows, ignore_index=True)

    molten = melted_df

    errors = []
    for _, row in molten.iterrows():
        errors.append(compute_dimensional_error(row.to_dict(), num_channels=1))
    molten["dimensional_error"] = errors

    flow_rates_per_batch = molten.groupby("batch_id")["flow_rate"].apply(list)
    cv_values = []
    recovery_values = []
    for _, row in molten.iterrows():
        batch_id = row["batch_id"]
        fr_list = flow_rates_per_batch.get(batch_id, [])
        fr_clean = []
        for fr in fr_list:
            try:
                if pd.isna(fr):
                    continue
                fr_val = float(fr)
                fr_clean.append(fr_val)
            except (TypeError, ValueError):
                continue
        fr_clean = [fr for fr in fr_clean if not pd.isna(fr)]
        if len(fr_clean) > 1:
            cv_values.append(compute_flow_rate_cv(fr_clean))
        else:
            cv_values.append(0.0)
        flow_rate_val = row.get("flow_rate", 0.0)
        try:
            if pd.isna(flow_rate_val):
                flow_rate_m3s = 0.0
            else:
                flow_rate_m3s = float(flow_rate_val) / (1e6 * 60)
        except (TypeError, ValueError):
            flow_rate_m3s = 0.0
        recovery_values.append(calculate_functional_recovery(flow_rate_m3s))
    molten["flow_rate_cv"] = cv_values
    molten["functional_recovery_error"] = recovery_values

    return molten


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
        df = pd.read_csv(path)
    else:
        df = pullData(sheet_name=sheet_name, verbose=verbose)

    melted_df = melt_dataset_to_single_channel(df, original_num_channels=4)

    if verbose:
        print(
            f"Data Melted: Converted {len(df)} batches into {len(melted_df)} independent channel trials."
        )

    return melted_df


def fake_objective(params: dict, context: dict, noise_std: float = 1.0) -> float:
    """
    Fake objective for testing only.
    """
    return float(np.random.normal(1e-6, noise_std))


def generate_random_deltas(mode: str = "uniform") -> dict:
    """
    Generate random post-print deltas using uniform distribution.

    Args:
        mode: Distribution type (currently only uniform supported)

    Returns:
        dict with length, width, height delta values in μm
    """
    return {
        "length": np.random.uniform(0, 20),
        "width": np.random.uniform(0, 20),
        "height": np.random.uniform(0, 20),
    }


def generate_realistic_deltas() -> dict:
    """
    Generate realistic post-print deltas combining:
    - Gaussian core (normal process variation, μ=8μm, σ=2.5μm)
    - Positive skew (overcure effect)
    - Occasional outliers (printer failures)

    Returns:
        dict with length, width, height delta values in μm
    """
    deltas = {}
    for dim in ["length", "width", "height"]:
        core = np.random.normal(8.0, 2.5)

        overcure = 0.0
        if np.random.random() < 0.3:
            overcure = np.random.uniform(0, 5)

        outlier = 0.0
        if np.random.random() < 0.05:
            outlier = np.random.uniform(15, 25)

        deltas[dim] = core + overcure + outlier

    return deltas


def simulate_print_trial(suggested_params: dict, delta_mode: str = "realistic") -> dict:
    """
    Simulate a print trial by generating post-print deltas and computing metrics.

    Args:
        suggested_params: Geometry parameters from CBO suggestion
        delta_mode: "random" for uniform, "realistic" for Gaussian+skew+outliers

    Returns:
        dict with all metrics including dimensional_error, flow_rate, flow_rate_cv
    """
    if delta_mode == "random":
        deltas = generate_random_deltas()
    else:
        deltas = generate_realistic_deltas()

    trial_params = {
        "channel_1_length": suggested_params.get("channel_1_length", 40.0),
        "channel_1_width": suggested_params.get("channel_1_width", 0.5),
        "channel_1_height": suggested_params.get("channel_1_height", 0.5),
        "channel_1_post_print_length_delta": deltas["length"],
        "channel_1_post_print_width_delta": deltas["width"],
        "channel_1_post_print_height_delta": deltas["height"],
    }

    error = compute_dimensional_error(trial_params, num_channels=1)

    flow_rate = np.random.normal(0.1387, 0.005)
    flow_rate_m3s = flow_rate / (1e6 * 60)
    recovery_error = calculate_functional_recovery(flow_rate_m3s)

    return {
        "dimensional_error": error,
        "flow_rate": flow_rate,
        "flow_rate_cv": 0.0,
        "functional_recovery_error": recovery_error,
        "deltas": deltas,
    }


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


def run_fake_trial(cbo, trial, context, delta_mode: str = "realistic"):
    """
    Run a fake trial for testing purposes.
    Generates post-print deltas, computes metrics, and observes trial.
    """
    suggested_params = trial.arms[0].parameters
    results = simulate_print_trial(suggested_params, delta_mode=delta_mode)

    print(
        f"  Generated deltas: L={results['deltas']['length']:.2f}, W={results['deltas']['width']:.2f}, H={results['deltas']['height']:.2f} μm"
    )
    print(f"  Dimensional Error: {results['dimensional_error']:.4f}")
    print(f"  Flow Rate: {results['flow_rate']:.4f} mL/min")
    print(f"  Functional Recovery Error: {results['functional_recovery_error']:.2f}%")

    cbo.observe(
        trial=trial,
        metric_values={
            "dimensional_error": results["dimensional_error"],
            "flow_rate": results["flow_rate"],
            "functional_recovery_error": results["functional_recovery_error"],
        }
    )


# returns (bool, dict | None) to pass warp deltas forward to next trial
def run_real_trial(
    trial, context, sheet_name: str, num_channels: int = NUM_CHANNELS
) -> tuple[bool, dict | None, int]:
    """
    Run a real trial: save suggested params to JSON and append to Google Sheets.
    Generates deltas based on sheet_name type.

    Args:
        sheet_name: Name of the Google Sheets tab to write to

    Returns:
        bool: True if trial completed successfully, False otherwise
        dict: warp deltas to pass forward to next trial
        int: batch_id for reference
    """
    suggested_params = trial.arms[0].parameters

    batch_raw = get_latest_col_value(column_name="batch_id", sheet_name=sheet_name)
    batch_id = int(batch_raw) if batch_raw is not None else 1
    batch_id += 1

    save_params_to_json(suggested_params, batch_id=batch_id)

    if "Realistic" in sheet_name:
        deltas = generate_realistic_deltas()
        delta_mode = "realistic"
    else:
        deltas = generate_random_deltas()
        delta_mode = "random"

    print(
        f"  Generated deltas ({delta_mode}): L={deltas['length']:.2f}, W={deltas['width']:.2f}, H={deltas['height']:.2f} μm"
    )

    params_with_deltas = {**suggested_params, **deltas}

    append_row(batch_id, params_with_deltas, context, sheet_name=sheet_name)
    print(f"\nAppended batch {batch_id} to Google Sheets.")

    trial_results = simulate_print_trial(suggested_params, delta_mode=delta_mode)
    print(f"  Dimensional Error: {trial_results['dimensional_error']:.4f}")
    print(f"  Flow Rate: {trial_results['flow_rate']:.4f} mL/min")
    print(
        f"  Functional Recovery Error: {trial_results['functional_recovery_error']:.2f}%"
    )

    warp_deltas = {}
    for i in range(1, num_channels + 1):
        for dim in ("length", "width", "height"):
            warp_deltas[f"channel_{i}_post_print_{dim}_delta"] = deltas[dim]

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
    Plot optimization convergence trace for all metrics:
    - Dimensional Error (primary)
    - Flow Rate
    - Functional Recovery Error
    """
    experiment = cbo.experiment
    data = experiment.fetch_data()
    if data is None or len(data.df) == 0:
        print("No data to visualize.")
        return

    df = data.df

    trial_indices = sorted(df["trial_index"].unique())
    if len(trial_indices) == 0:
        print("No trials to visualize.")
        return

    dimensional_errors = []
    flow_rates = []
    functional_errors = []

    for tri_idx in trial_indices:
        tri_df = df[df["trial_index"] == tri_idx]

        de_row = tri_df[tri_df["metric_name"] == "dimensional_error"]
        fr_row = tri_df[tri_df["metric_name"] == "flow_rate"]
        fre_row = tri_df[tri_df["metric_name"] == "functional_recovery_error"]

        dimensional_errors.append(de_row["mean"].iloc[0] if len(de_row) > 0 else 0.0)
        flow_rates.append(fr_row["mean"].iloc[0] if len(fr_row) > 0 else 0.0)
        functional_errors.append(fre_row["mean"].iloc[0] if len(fre_row) > 0 else 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(trial_indices, dimensional_errors, "b-o", linewidth=2, markersize=6)
    axes[0].set_title("Dimensional Error (MSE) - Primary Objective", fontsize=12)
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Dimensional Error (μm²)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(trial_indices, flow_rates, "g-o", linewidth=2, markersize=6)
    axes[1].axhline(
        y=0.1387, color="r", linestyle="--", label="Baseline (0.1387 mL/min)"
    )
    axes[1].set_title("Flow Rate Over Trials", fontsize=12)
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Flow Rate (mL/min)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(trial_indices, functional_errors, "m-o", linewidth=2, markersize=6)
    axes[2].axhline(y=5.0, color="r", linestyle="--", label="5% Threshold")
    axes[2].set_title("Functional Recovery Error (% from baseline)", fontsize=12)
    axes[2].set_xlabel("Trial")
    axes[2].set_ylabel("Recovery Error (%)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
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
            metric_name="dimensional_error",
            minimize=True,
            tracking_metrics=["flow_rate", "flow_rate_cv", "functional_recovery_error"],
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
            completed, prev_warp, batch_id, trial_results = run_real_trial(trial, context, sheet_name)
            if not completed:
                return
            
            cbo.observe(
                trial=trial,
                metric_values={
                    "dimensional_error": trial_results["dimensional_error"],
                    "flow_rate": trial_results["flow_rate"],
                    "functional_recovery_error": trial_results["functional_recovery_error"],
                }
            )

            print(
                f"\nBatch {batch_id} saved. CBO updated with dimensional_error: {trial_results['dimensional_error']:.4f}"
            )
        else:
            run_fake_trial(cbo, trial, context)
            prev_warp = None

        visualize_convergence(cbo)

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
