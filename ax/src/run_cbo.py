import json
from config import (
    NUM_CHANNELS,
    CHANNEL_LENGTH_BOUNDS,
    CHANNEL_WIDTH_BOUNDS,
    CHANNEL_HEIGHT_BOUNDS,
    MIN_CHANNEL_SPACING,
    BASE_WIDTH,
)
from ax_cbo import ContextualBayesOptAx
from ax.core import SearchSpace, RangeParameter, ChoiceParameter, ParameterType
from ax.core.parameter_constraint import ParameterConstraint
import numpy as np


def build_search_space(num_channels: int = NUM_CHANNELS):
    """
    Build search space with independent parameters for each channel.

    Parameters:
        num_channels: Number of independent channels (default: 4)

    Returns:
        SearchSpace with:
        - 3 × num_channels channel dimension parameters
        - 1 layer thickness parameter
        - 3 context parameters (ambient_temp, resin_temp, resin_age)
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


def estimate_hydraulic_resistance(length: float, width: float, height: float) -> float:
    """
    Estimate hydraulic resistance using Hagen-Poiseuille approximation for rectangular channels.

    For rectangular cross-section, R ~ (12 * mu * L) / (w * h^3) for h << w
    Using simplified form: R proportional to L / (w * h^3)

    Args:
        length: Channel length (mm)
        width: Channel width (mm)
        height: Channel height (mm)

    Returns:
        Relative hydraulic resistance (arbitrary units)
    """
    # Avoid division by zero
    if width <= 0 or height <= 0:
        return float("inf")
    return length / (width * (height**3))


def run_cbo_for_cad():
    """
    Run contextual Bayesian optimization to suggest parameters for 4 independent channels.

    Optimizes for uniform flow (minimizes CV of flow rates across channels).
    """

    context = {
        "ambient_temp": 80.0,
        "resin_temp": 80.0,
        "resin_age": 15.0,
    }

    cbo = ContextualBayesOptAx(
        search_space=build_search_space(),
        metric_name="flow_rate_cv",  # Coefficient of variation - minimize for uniformity
        minimize=True,
    )

    # Skip historical data - dataset doesn't have per-channel parameters yet
    # Future: load CFD simulation results here
    # df = pd.read_csv("path/to/cfd_results.csv")
    # cbo.add_historical(df)
    print("Running CBO (cold start) for 4 independent channels")
    print("Optimizing for flow rate uniformity (minimize CV)")

    result = cbo.suggest(isOnline=False, c_t=context)
    trial = result["trial"]
    suggested_params = trial.arms[0].parameters

    # Extract per-channel parameters for clarity
    channels = extract_channel_params(suggested_params)

    print("\nSuggested parameters:")
    for i, ch in enumerate(channels, 1):
        print(
            f"  Channel {i}: L={ch['length']:.2f}, W={ch['width']:.2f}, H={ch['height']:.2f} mm"
        )
    print(f"  Layer thickness: {suggested_params['layer_thickness_um']} um")

    # Estimate relative hydraulic resistances (for sanity check before CFD)
    resistances = [
        estimate_hydraulic_resistance(ch["length"], ch["width"], ch["height"])
        for ch in channels
    ]
    print("\nEstimated relative hydraulic resistances:")
    for i, r in enumerate(resistances, 1):
        print(f"  Channel {i}: {r:.4f}")
    print(f"  Resistance CV: {compute_flow_rate_cv(resistances):.4f}")

    return suggested_params


if __name__ == "__main__":
    params = run_cbo_for_cad()
    with open("suggested_params.json", "w") as f:
        json.dump(params, f)
    print("Saved to suggested_params.json")
