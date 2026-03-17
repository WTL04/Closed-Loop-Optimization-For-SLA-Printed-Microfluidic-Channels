import cadquery as cq
import numpy as np

# Configuration constants
NUM_CHANNELS = 4
MIN_CHANNEL_SPACING = 2.0  # mm - minimum gap between adjacent channel rows

# Channel dimension bounds (same for all channels)
CHANNEL_LENGTH_BOUNDS = (30.0, 90.0)  # mm
CHANNEL_WIDTH_BOUNDS = (10.0, 30.0)  # mm
CHANNEL_HEIGHT_BOUNDS = (10.0, 30.0)  # mm

# Base plate dimensions for CAD model
BASE_LENGTH = 120.0  # mm (X direction)
BASE_WIDTH = 100.0  # mm (Y direction - channels arranged along this axis)
BASE_THICKNESS = 2.0  # mm


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

    # Lazy import
    from ax_cbo import ContextualBayesOptAx
    from ax.core import SearchSpace, RangeParameter, ChoiceParameter, ParameterType
    from ax.core.parameter_constraint import ParameterConstraint

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

        # Channel Z position (sitting on top of base)
        channel_center_z = BASE_THICKNESS / 2 + ch["height"] / 2

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


if __name__ == "__main__":
    params = run_cbo_for_cad()
    model = build_cad_model(params)

    cq.show_object(model)

    # print("\nExporting to STL...")
    # cq.exporters.export(model, "../cad_models/channel_test.stl")
    # print("Exported to ../cad_models/channel_test.stl")

    # Print summary for CAD/CFD reference
    channels = extract_channel_params(params)
    print("\n" + "=" * 50)
    print("PARAMETERS FOR CFD SIMULATION")
    print("=" * 50)
    for i, ch in enumerate(channels, 1):
        print(f"Channel {i}:")
        print(f"  Length: {ch['length']:.2f} mm")
        print(f"  Width:  {ch['width']:.2f} mm")
        print(f"  Height: {ch['height']:.2f} mm")
    print(f"\nLayer thickness: {params['layer_thickness_um']} um")
    print(f"Channel spacing: {MIN_CHANNEL_SPACING} mm")
    print("=" * 50)
