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
    Compute mean squared error between the fabricated dimensions and the
    nominal target dimensions (40 x 0.5 x 0.5 mm).

    Uses exact column names from dataset:
    - channel_length_mm, channel_width_mm, channel_height_mm (mm)
    - delta_length_um, delta_width_um, delta_height_um (um)

    Args:
        params: Dict with channel_length_mm/channel_width_mm/channel_height_mm (mm) and delta_*_um (µm)
        num_channels: Number of channels (default: NUM_CHANNELS from config)

    Returns:
        Mean squared error in mm^2 (lower is better)
    """
    # Use exact column names from dataset
    length = params.get("channel_length_mm", NOMINAL_DIMENSIONS["length"])
    width = params.get("channel_width_mm", NOMINAL_DIMENSIONS["width"])
    height = params.get("channel_height_mm", NOMINAL_DIMENSIONS["height"])
    delta_length = params.get("delta_length_um", 0.0) or 0.0
    delta_width = params.get("delta_width_um", 0.0) or 0.0
    delta_height = params.get("delta_height_um", 0.0) or 0.0
    
    try:
        length_delta_mm = float(delta_length) / 1000.0
        width_delta_mm = float(delta_width) / 1000.0
        height_delta_mm = float(delta_height) / 1000.0
    except (TypeError, ValueError):
        length_delta_mm = 0.0
        width_delta_mm = 0.0
        height_delta_mm = 0.0

    fabricated_length = float(length) - length_delta_mm
    fabricated_width = float(width) - width_delta_mm
    fabricated_height = float(height) - height_delta_mm

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