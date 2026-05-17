"""
Runner for executing CBO trials.

Functions:
- run_single_channel: Execute single CBO trial
"""

import numpy as np

from contextual_opt.src.pipeline.config import NOMINAL_DIMENSIONS
from contextual_opt.src.pipeline.metrics import compute_dimensional_error
from contextual_opt.src.pipeline.delta_loaders import (
    generate_random_deltas,
    generate_realistic_deltas,
)
from contextual_opt.src.pipeline.cfd_runs import run_cfd_simulation


def run_single_channel(
    cbo, context, sheet_name: str, channel_num: int, use_real_data: bool,
    case_dir: str = "cfd/channelCase",
):
    """
    Run CBO for a SINGLE channel, return result.

    Args:
        cbo: ContextualBayesOptAx instance
        context: Context dict with layer_thickness_um, ambient_temp, resin_temp, resin_age
        sheet_name: Name of the Google Sheet tab
        channel_num: Channel number (1-4)
        use_real_data: If True, run CFD; if False, use fake data

    Returns:
        dict with channel results including dim_error and flow_rate
    """
    channel_context = dict(context)

    result = cbo.suggest(c_t=channel_context)
    trial = result["trial"]
    suggested_params = trial.arm.parameters

    channel_length_um = suggested_params.get(
        "channel_length_um", NOMINAL_DIMENSIONS["length"]
    )
    channel_width_um = suggested_params.get(
        "channel_width_um", NOMINAL_DIMENSIONS["width"]
    )
    channel_height_um = suggested_params.get(
        "channel_height_um", NOMINAL_DIMENSIONS["height"]
    )

    print(f"\n=== Channel {channel_num} ===")
    print(f"Length: {channel_length_um:.1f} µm")
    print(f"Width: {channel_width_um:.1f} µm")
    print(f"Height: {channel_height_um:.1f} µm")
    print(f"Layer Thickness: {suggested_params.get('layer_thickness_um', 'N/A')} µm")

    if not use_real_data:
        # Fake trial
        flow_rate = np.random.uniform(0.1, 0.2)
        deltas = generate_random_deltas()
        channel_results = {
            "channel": channel_num,
            "channel_length_um": channel_length_um,
            "channel_width_um": channel_width_um,
            "channel_height_um": channel_height_um,
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
            cbo_length_um=channel_length_um,
            cbo_width_um=channel_width_um,
            cbo_height_um=channel_height_um,
            length_delta=deltas["length"],
            width_delta=deltas["width"],
            height_delta=deltas["height"],
            case_dir=case_dir,
        )
        flow_rate_ml = flow_rate_m3s * 1e6 * 60

        channel_results = {
            "channel": channel_num,
            "channel_length_um": channel_length_um,
            "channel_width_um": channel_width_um,
            "channel_height_um": channel_height_um,
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
    metric_values = {"dim_error": dim_error}
    flow_rate = channel_results["flow_rate"]
    if flow_rate > 0:
        metric_values["flow_rate"] = flow_rate
    elif flow_rate_m3s == -1.0:
        print(
            "  WARNING: CFD simulation failed (sentinel -1.0) — skipping flow_rate for surrogate"
        )
    else:
        print(
            "  WARNING: CFD returned zero flow rate — skipping flow_rate for surrogate"
        )

    cbo.observe(trial=trial, metric_values=metric_values)

    return channel_results
