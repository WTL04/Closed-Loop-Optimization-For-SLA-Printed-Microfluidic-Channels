"""
Computation functions for dimensional and flow rate calculations.

Functions:
- compute_flow_rate_cv
- compute_dimensional_error
- calculate_functional_recovery
"""

import numpy as np

from contextual_opt.src.pipeline.config import (
    NUM_CHANNELS,
    BASELINE_FLOW_RATE,
    NOMINAL_DIMENSIONS,
)


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
    Compute MSE between the fabricated dimensions (Ax suggested) and the
    nominal target dimensions (40000 x 500 x 500 µm).

    Uses exact column names from dataset:
    - channel_length_um, channel_width_um, channel_height_um (µm)
    - delta_length_um, delta_width_um, delta_height_um (µm)

    Args:
        params: Dict with channel_length_um/channel_width_um/channel_height_um (µm) and delta_*_um (µm)
        num_channels: Number of channels (default: NUM_CHANNELS from config)

    Returns:
        Mean squared error in µm^2 (lower is better)
    """
    # Use exact column names from dataset
    length = params.get("channel_length_um", NOMINAL_DIMENSIONS["length"])
    width = params.get("channel_width_um", NOMINAL_DIMENSIONS["width"])
    height = params.get("channel_height_um", NOMINAL_DIMENSIONS["height"])
    delta_length = params.get("delta_length_um", 0.0) or 0.0
    delta_width = params.get("delta_width_um", 0.0) or 0.0
    delta_height = params.get("delta_height_um", 0.0) or 0.0

    try:
        # deltas are in µm
        length_delta = float(delta_length)
        width_delta = float(delta_width)
        height_delta = float(delta_height)
    except (TypeError, ValueError):
        length_delta = 0.0
        width_delta = 0.0
        height_delta = 0.0

    # fabricated dimensions = ax suggested - random deltas
    fabricated_length = float(length) - length_delta
    fabricated_width = float(width) - width_delta
    fabricated_height = float(height) - height_delta

    # calculate MSE
    squared_errors = [
        (fabricated_length - NOMINAL_DIMENSIONS["length"]) ** 2,
        (fabricated_width - NOMINAL_DIMENSIONS["width"]) ** 2,
        (fabricated_height - NOMINAL_DIMENSIONS["height"]) ** 2,
    ]

    return float(np.mean(squared_errors)) if squared_errors else 0.0


def calculate_functional_recovery(extracted_flow_rate: float) -> float:
    """
    Calculates how close the pre-distorted channel's flow rate
    is to the perfect nominal baseline.

    Args:
        extracted_flow_rate: Flow rate in m^3/s from OpenFOAM

    Returns:
        Error percentage (0.0 = perfect recovery)
    """
    return abs(extracted_flow_rate - BASELINE_FLOW_RATE) / BASELINE_FLOW_RATE * 100.0

