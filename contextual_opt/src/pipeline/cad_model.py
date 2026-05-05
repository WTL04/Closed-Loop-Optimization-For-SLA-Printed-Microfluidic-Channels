"""
CAD model building and export functions.

Functions:
- build_cad_model
- export_cad_model
"""
import cadquery as cq
from pathlib import Path

from contextual_opt.src.pipeline.config import (
    NUM_CHANNELS,
    MIN_CHANNEL_SPACING,
    BASE_LENGTH,
    BASE_WIDTH,
    BASE_THICKNESS,
)


def build_cad_model(params: dict, num_channels: int = NUM_CHANNELS):
    """
    Build CAD model with N independent channel rows.

    Channels are arranged as parallel rows along the Y-axis with minimum spacing.

    Args:
        params: Parameter dict with length/width/height for each channel
        num_channels: Number of channels

    Returns:
        CadQuery Workplane object ready for export
    """
    channel_length = params.get("length", 40.0)
    channel_width = params.get("width", 0.5)
    channel_height = params.get("height", 0.5)

    total_spacing = (num_channels - 1) * MIN_CHANNEL_SPACING
    available_y = BASE_WIDTH - total_spacing - (num_channels * channel_width)
    start_y = available_y / 2 if available_y > 0 else 0

    result = cq.Workplane("XY")

    for i in range(num_channels):
        y_pos = start_y + i * (channel_width + MIN_CHANNEL_SPACING)

        channel = (
            cq.Workplane("XY")
            .box(channel_length, channel_width, channel_height)
            .translate((0, y_pos + channel_width / 2, channel_height / 2))
        )
        result = result.union(channel)

    base = cq.Workplane("XY").box(
        BASE_LENGTH, BASE_WIDTH, BASE_THICKNESS
    ).translate((0, 0, -BASE_THICKNESS / 2))

    result = result.cut(base)

    return result


def export_cad_model(params: dict, filename: str = "contextual_opt/cad_models/channels_fluid.stl"):
    """
    Export CAD model to STL file.

    Args:
        params: Parameter dict with length/width/height
        filename: Output STL filename
    """
    model = build_cad_model(params)

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    model.export(filename)
    print(f"Exported CAD model to {filename}")