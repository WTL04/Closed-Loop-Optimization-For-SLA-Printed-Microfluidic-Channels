import cadquery as cq
from cadquery import Face, Compound
import os
import sys

# CFD extensions (mm)
INLET_LENGTH = 5.0
OUTLET_LENGTH = 5.0

# Shift to keep geometry inside blockMesh background mesh
X_SHIFT = 0.0

# Paths
STL_EXPORT_DIR = "cfd/channelCase/constant/triSurface/"
CFD_RUN_SCRIPT = "run_cfd.sh"

SHEET_NAME = "Experiment Realistic Deltas"


def compute_expected_physical_dimensions(
    cbo_l_um, cbo_w_um, cbo_h_um, delta_l_um, delta_w_um, delta_h_um
):
    """
    Calculates the actual physical dimensions that will exit the 3D printer.
    Expected Physical = CBO_Suggested_CAD_Input + Printer_Delta_Error
    Converts from micrometers (um) to millimeters (mm) for CadQuery.
    """
    phys_l_um = cbo_l_um - delta_l_um
    phys_w_um = cbo_w_um - delta_w_um
    phys_h_um = cbo_h_um - delta_h_um

    L = phys_l_um / 1000.0
    W = phys_w_um / 1000.0
    H = phys_h_um / 1000.0

    if L <= 0 or W <= 0 or H <= 0:
        raise ValueError(
            f"Physical dimensions became non-positive: L={L}, W={W}, H={H}. "
            f"CBO inputs (um): l={cbo_l_um}, w={cbo_w_um}, h={cbo_h_um}; "
            f"Deltas (um): l={delta_l_um}, w={delta_w_um}, h={delta_h_um}."
        )
    return L, W, H


def build_channel(cbo_l_um, cbo_w_um, cbo_h_um, delta_l_um, delta_w_um, delta_h_um):
    L, W, H = compute_expected_physical_dimensions(
        cbo_l_um, cbo_w_um, cbo_h_um, delta_l_um, delta_w_um, delta_h_um
    )
    total_length = INLET_LENGTH + L + OUTLET_LENGTH

    # World-space X coordinates of channel ends
    x_start = X_SHIFT  # left end (inlet face)
    x_end = X_SHIFT + total_length  # right end (outlet face)
    x_mid = X_SHIFT + total_length / 2.0  # centre for box placement
    y_mid = 0.0
    z_mid = H / 2.0

    # --- Fluid volume: full enclosed rectangular prism ---
    fluid = cq.Workplane("XY").box(total_length, W, H).translate((x_mid, y_mid, z_mid))

    # --- Inlet: flat face at x_start (thin extrusion for STL export) ---
    # Uses YZ plane so the face normal points along X
    inlet = (
        cq.Workplane("YZ")
        .rect(W, H)
        .extrude(0.01)  # 10 um thickness -- thick enough for snappyHexMesh to resolve
        .translate((x_start, y_mid, z_mid))
    )

    # --- Outlet: flat face at x_end ---
    outlet = (
        cq.Workplane("YZ")
        .rect(W, H)
        .extrude(0.01)
        .translate((x_end - 0.01, y_mid, z_mid))
    )

    # --- Walls: 4 long faces only (top, bottom, left, right) ---
    # Get all faces from the box solid, filter out the two X-axis end faces
    # by checking the face normal vector's X component.
    # End faces have normal.x == +/-1.0, wall faces have normal.x == 0.0
    wall_box = (
        cq.Workplane("XY").box(total_length, W, H).translate((x_mid, y_mid, z_mid))
    )

    # .vals() returns the underlying OCCT Face objects
    # Face.normalAt() returns a Vector -- access .x directly
    wall_faces: list[Face] = [
        f
        for f in wall_box.faces().vals()
        if isinstance(f, Face) and abs(f.normalAt(f.Center()).x) < 0.99
    ]

    if len(wall_faces) != 4:
        raise RuntimeError(
            f"Expected 4 wall faces, got {len(wall_faces)}. "
            f"Check geometry -- box may be degenerate."
        )

    walls = Compound.makeCompound(wall_faces)

    return fluid, walls, inlet, outlet, L, W, H


def run_pipeline(cbo_l_um, cbo_w_um, cbo_h_um, delta_l_um, delta_w_um, delta_h_um):
    """
    Pipeline that creates a channel given Ax CBO's pre-distortion dimensions, and post print deltas
    """

    fluid, walls, inlet, outlet, L, W, H = build_channel(
        cbo_l_um, cbo_w_um, cbo_h_um, delta_l_um, delta_w_um, delta_h_um
    )

    os.makedirs(STL_EXPORT_DIR, exist_ok=True)

    exports = {
        "channels_fluid.stl": fluid,
        "channel_walls.stl": walls,
        "channel_inlet.stl": inlet,
        "channel_outlet.stl": outlet,
    }

    for filename, shape in exports.items():
        path = os.path.join(STL_EXPORT_DIR, filename)
        cq.exporters.export(shape, path)
        print(f"Exported {filename}")

    print(f"\nChannel dimensions: L={L:.4f} mm, W={W:.4f} mm, H={H:.4f} mm")
    print(
        f"Total CFD length (with extensions): {INLET_LENGTH + L + OUTLET_LENGTH:.4f} mm"
    )


if __name__ == "__main__":
    if len(sys.argv) >= 7:
        cbo_l = float(sys.argv[1])
        cbo_w = float(sys.argv[2])
        cbo_h = float(sys.argv[3])
        delta_l = float(sys.argv[4])
        delta_w = float(sys.argv[5])
        delta_h = float(sys.argv[6])
    elif len(sys.argv) >= 4:
        print("Received 3 args — treating as deltas only, using nominal CBO inputs.")
        cbo_l = 40000.0
        cbo_w = 500.0
        cbo_h = 500.0
        delta_l = float(sys.argv[1])
        delta_w = float(sys.argv[2])
        delta_h = float(sys.argv[3])
    else:
        try:
            from contextual_opt.src import sheets_api

            print(
                f"No arguments provided. Fetching latest deltas from '{SHEET_NAME}'..."
            )
            cols = {
                "length": "channel_1_post_print_length_delta",
                "width": "channel_1_post_print_width_delta",
                "height": "channel_1_post_print_height_delta",
            }
            delta_l = float(
                sheets_api.get_latest_col_value(cols["length"], sheet_name=SHEET_NAME)
                or 0.0
            )
            delta_w = float(
                sheets_api.get_latest_col_value(cols["width"], sheet_name=SHEET_NAME)
                or 0.0
            )
            delta_h = float(
                sheets_api.get_latest_col_value(cols["height"], sheet_name=SHEET_NAME)
                or 0.0
            )
        except Exception as e:
            print(f"Could not fetch from sheets: {e}. Using zero deltas.")
            delta_l, delta_w, delta_h = 0.0, 0.0, 0.0

        print("No CBO args provided. Using nominal CBO inputs (40000, 500, 500 um).")
        cbo_l, cbo_w, cbo_h = 40000.0, 500.0, 500.0

    run_pipeline(cbo_l, cbo_w, cbo_h, delta_l, delta_w, delta_h)
