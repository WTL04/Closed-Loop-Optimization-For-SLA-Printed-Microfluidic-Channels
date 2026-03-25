import cadquery as cq
import json

from config import (
    NUM_CHANNELS,
    MIN_CHANNEL_SPACING,
)


# -----------------------------
# User-defined CFD extension lengths (mm)
# -----------------------------
INLET_LENGTH = 5.0
OUTLET_LENGTH = 5.0


def extract_channel_params(
    params: dict, num_channels: int = NUM_CHANNELS
) -> list[dict]:
    """
    Extract individual channel parameters from a flat parameter dictionary.

    Expected keys:
        channel_1_length, channel_1_width, channel_1_height, ...
    """
    channels = []
    for i in range(1, num_channels + 1):
        channels.append(
            {
                "length": float(params[f"channel_{i}_length"]),
                "width": float(params[f"channel_{i}_width"]),
                "height": float(params[f"channel_{i}_height"]),
            }
        )
    return channels


def build_fluid_domain(params: dict, num_channels: int = NUM_CHANNELS):
    """
    Build CFD-ready fluid-domain geometry for independent channels.

    Each channel is modeled as:
        [inlet extension] + [main channel] + [outlet extension]

    This returns only the fluid volume, not the surrounding solid chip.
    """

    channels = extract_channel_params(params, num_channels)

    # Compute overall Y layout
    total_width = sum(ch["width"] for ch in channels)
    total_spacing = (num_channels - 1) * MIN_CHANNEL_SPACING
    y_start = -(total_width + total_spacing) / 2.0

    result = None
    current_y = y_start

    for ch in channels:
        L = ch["length"]
        W = ch["width"]
        H = ch["height"]

        # Center position in Y for this channel
        channel_center_y = current_y + W / 2.0

        # Put channel bottom on z = 0
        channel_center_z = H / 2.0

        # Total length of one flow path
        total_flow_length = INLET_LENGTH + L + OUTLET_LENGTH

        # Build as one long rectangular fluid body
        full_channel = (
            cq.Workplane("XY")
            .box(total_flow_length, W, H)
            .translate((0.0, channel_center_y, channel_center_z))
        )

        # Union all channels into one export object
        if result is None:
            result = full_channel
        else:
            result = result.union(full_channel)

        current_y += W + MIN_CHANNEL_SPACING

    return result


if __name__ == "__main__":
    with open("suggested_params.json", "r") as f:
        params = json.load(f)

    model = build_fluid_domain(params)
    cq.exporters.export(model, "channels_fluid.stl")

    print("CFD fluid-domain STL generated: channels_fluid.stl")
    print(f"Inlet length  = {INLET_LENGTH} mm")
    print(f"Outlet length = {OUTLET_LENGTH} mm")