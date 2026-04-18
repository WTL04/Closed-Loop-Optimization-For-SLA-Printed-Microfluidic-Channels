import cadquery as cq
import json

# Fixed nominal dimensions for every channel (mm)
BASE_LENGTH = 40.0
BASE_WIDTH = 0.5
BASE_HEIGHT = 0.5

# Extra CFD extensions (mm)
INLET_LENGTH = 5.0
OUTLET_LENGTH = 5.0

# Shift channel so it stays inside blockMesh domain
X_SHIFT = 30.0


def get_post_print_dimensions(params: dict, channel_index: int = 1) -> dict:
    """
    Compute post-print dimensions using:
        new = old + delta

    Here old dimensions are fixed for every channel:
        L = 40 mm
        W = 0.5 mm
        H = 0.5 mm
    """

    length_delta = float(params.get(f"channel_{channel_index}_post_print_length_delta", 0.0))
    width_delta = float(params.get(f"channel_{channel_index}_post_print_width_delta", 0.0))
    height_delta = float(params.get(f"channel_{channel_index}_post_print_height_delta", 0.0))

    post_length = BASE_LENGTH + length_delta
    post_width = BASE_WIDTH + width_delta
    post_height = BASE_HEIGHT + height_delta

    if post_length <= 0 or post_width <= 0 or post_height <= 0:
        raise ValueError(
            f"Non-positive post-print dimensions for channel {channel_index}: "
            f"L={post_length}, W={post_width}, H={post_height}"
        )

    return {
        "length": post_length,
        "width": post_width,
        "height": post_height,
    }


def build_single_channel_parts(params: dict, channel_index: int = 1):
    """
    Build separate CAD bodies for:
    - full fluid volume
    - walls
    - inlet face
    - outlet face
    """

    ch = get_post_print_dimensions(params, channel_index=channel_index)

    L = ch["length"]
    W = ch["width"]
    H = ch["height"]

    total_flow_length = INLET_LENGTH + L + OUTLET_LENGTH

    fluid = (
        cq.Workplane("XY")
        .box(total_flow_length, W, H)
        .translate((X_SHIFT, 0.0, H / 2.0))
    )

    x_min = -total_flow_length / 2.0
    x_max = total_flow_length / 2.0

    # Thick enough for separate STL patch creation
    face_thickness = max(min(W, H) * 0.2, 0.05)

    inlet = (
        cq.Workplane("XY")
        .box(face_thickness, W, H)
        .translate((X_SHIFT + x_min + face_thickness / 2.0, 0.0, H / 2.0))
    )

    outlet = (
        cq.Workplane("XY")
        .box(face_thickness, W, H)
        .translate((X_SHIFT + x_max - face_thickness / 2.0, 0.0, H / 2.0))
    )

    walls = fluid.cut(inlet).cut(outlet)

    return fluid, walls, inlet, outlet, ch


if __name__ == "__main__":
    with open("suggested_params.json", "r") as f:
        params = json.load(f)

    channel_index = 1

    fluid, walls, inlet, outlet, dims = build_single_channel_parts(
        params, channel_index=channel_index
    )

    cq.exporters.export(fluid, "channels_fluid.stl")
    cq.exporters.export(walls, "channel_walls.stl")
    cq.exporters.export(inlet, "channel_inlet.stl")
    cq.exporters.export(outlet, "channel_outlet.stl")

    print("Generated:")
    print("  channels_fluid.stl")
    print("  channel_walls.stl")
    print("  channel_inlet.stl")
    print("  channel_outlet.stl")
    print()
    print("Post-print dimensions used:")
    print(f"  Length = {dims['length']} mm")
    print(f"  Width  = {dims['width']} mm")
    print(f"  Height = {dims['height']} mm")