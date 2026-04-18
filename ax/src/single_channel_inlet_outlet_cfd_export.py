import cadquery as cq

# Fixed nominal dimensions (mm)
BASE_LENGTH = 40.0
BASE_WIDTH = 0.5
BASE_HEIGHT = 0.5

# CFD extensions (mm)
INLET_LENGTH = 5.0
OUTLET_LENGTH = 5.0

# Shift to keep geometry inside mesh
X_SHIFT = 30.0


def get_user_deltas():
    print("\nEnter post-print deltas (in mm):")

    length_delta = float(input("channel post print length delta: "))
    width_delta = float(input("channel post print width delta: "))
    height_delta = float(input("channel post print height delta: "))

    return length_delta, width_delta, height_delta


def compute_post_print_dimensions():
    length_delta, width_delta, height_delta = get_user_deltas()

    L = BASE_LENGTH + length_delta
    W = BASE_WIDTH + width_delta
    H = BASE_HEIGHT + height_delta

    if L <= 0 or W <= 0 or H <= 0:
        raise ValueError("Post-print dimensions became non-positive!")

    return L, W, H


def build_channel():
    L, W, H = compute_post_print_dimensions()

    total_length = INLET_LENGTH + L + OUTLET_LENGTH

    fluid = (
        cq.Workplane("XY")
        .box(total_length, W, H)
        .translate((X_SHIFT, 0.0, H / 2.0))
    )

    x_min = -total_length / 2.0
    x_max = total_length / 2.0

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

    return fluid, walls, inlet, outlet, L, W, H


if __name__ == "__main__":
    fluid, walls, inlet, outlet, L, W, H = build_channel()

    cq.exporters.export(fluid, "channels_fluid.stl")
    cq.exporters.export(walls, "channel_walls.stl")
    cq.exporters.export(inlet, "channel_inlet.stl")
    cq.exporters.export(outlet, "channel_outlet.stl")

    print("\nGenerated STL files successfully!")

    print("\nPost-print dimensions used:")
    print(f"Length = {L} mm")
    print(f"Width  = {W} mm")
    print(f"Height = {H} mm")