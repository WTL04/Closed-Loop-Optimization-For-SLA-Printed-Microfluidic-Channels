import cadquery as cq
import json

from config import (
    NUM_CHANNELS,
    MIN_CHANNEL_SPACING,
    CHANNEL_LENGTH_BOUNDS,
    CHANNEL_WIDTH_BOUNDS,
    CHANNEL_HEIGHT_BOUNDS,
    BASE_LENGTH,
    BASE_WIDTH,
    BASE_THICKNESS,
)


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


if __name__ == "__main__":
    with open("suggested_params.json") as f:
        params = json.load(f)

    model = build_cad_model(params)

    user_input = input(
        "\n 1) Export CAD Model \n 2) View CAD model in cq-editor \n (Default: 1): "
    )

    if user_input == "1":
        cq.exporters.export(model, "../cad_models/channels_fluid.stl")
        print("CFD fluid-domain STL generated: channels_fluid.stl")
        print("Path to Model: ../cad_models/channel_test.stl")
    elif user_input == "2":
        show_object(model)
    else:
        raise ValueError("Invalid export/view option")
