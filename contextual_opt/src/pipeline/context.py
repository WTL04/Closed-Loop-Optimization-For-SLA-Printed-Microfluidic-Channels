"""
Context generation for CBO.

Functions:
- get_context_snapshot: Interactive context input
- context_overtime: Automated context generation with drift
"""

import numpy as np

from contextual_opt.src.pipeline.config import (
    AMBIENT_TEMP_BOUNDS,
    RESIN_TEMP_BOUNDS,
    RESIN_AGE_BOUNDS,
    CONTEXT_DRIFT_MAX,
    CONTEXT_NOISE_MAX,
)


def get_context_snapshot(prev_warp=None):
    """
    Get context snapshot for CBO (interactive).

    Args:
        prev_warp: Not currently used, kept for API compatibility

    Returns:
        dict with layer_thickness_um, ambient_temp, resin_temp, resin_age
    """
    print(
        "\n1) Manually input context snapshot \n2) Use fixed testing context snapshot"
    )
    choice = input("\nPlease choose one of the two: ")

    if choice == "1":
        layer_thickness = int(input("Layer thickness (µm): "))
        ambient = float(input("Ambient temperature (°F): "))
        resin = float(input("Resin temperature (°F): "))
        age = float(input("Resin age (hours): "))
        return {
            "layer_thickness_um": layer_thickness,
            "ambient_temp": ambient,
            "resin_temp": resin,
            "resin_age": age,
        }
    else:
        return {
            "layer_thickness_um": 100,
            "ambient_temp": 80.0,
            "resin_temp": 80.0,
            "resin_age": 15.0,
        }


def context_overtime(
    temp: str,
    layer_thickness_um: int,
    testing: bool = False,
    start_ambient: float = 80.0,
    start_resin_age: float = 1.0,
    resin_temp: float = 80.0,
):
    """
    Generate context that drifts across 4 channel trials.

    Args:
        temp: "cold" (ambient decreases) or "hot" (ambient increases)
        layer_thickness_um: 50 or 100 - decided by user at start
        testing: If True, use interactive prompts; if False, automated
        start_ambient: Starting ambient temp in F (default 80)
        start_resin_age: Starting resin age in hours (default 1)
        resin_temp: Base resin temp in F (default 80)

    Returns:
        List of 4 context dicts, one per channel trial
    """
    if testing:
        # Use interactive prompts (existing behavior)
        return [get_context_snapshot() for _ in range(4)]

    # Automated context generation
    contexts = []
    current_ambient = start_ambient
    resin_age = start_resin_age  # Start at provided resin age

    for ch in range(1, 5):
        # Calculate drift direction based on temp type
        if ch == 1:
            # First channel: small random from start
            drift = np.random.uniform(-CONTEXT_DRIFT_MAX / 2, CONTEXT_DRIFT_MAX / 2)
        else:
            # Subsequent channels: drift in direction of temp type
            drift = np.random.uniform(0, CONTEXT_DRIFT_MAX)

        if temp == "cold":
            ambient = current_ambient - drift
        else:  # hot
            ambient = current_ambient + drift

        # Add small random noise
        noise = np.random.uniform(-CONTEXT_NOISE_MAX, CONTEXT_NOISE_MAX)
        ambient += noise

        # Clamp to bounds
        ambient = max(AMBIENT_TEMP_BOUNDS[0], min(AMBIENT_TEMP_BOUNDS[1], ambient))

        # Resin temp with small noise (constant + noise)
        resin = resin_temp + np.random.uniform(-1, 1)
        resin = max(RESIN_TEMP_BOUNDS[0], min(RESIN_TEMP_BOUNDS[1], resin))

        context = {
            "layer_thickness_um": layer_thickness_um,
            "ambient_temp": round(ambient, 1),
            "resin_temp": round(resin, 1),
            "resin_age": resin_age,
        }
        contexts.append(context)

        # Update for next channel
        current_ambient = ambient
        resin_age += 6  # Increase by 6 hours

    return contexts
