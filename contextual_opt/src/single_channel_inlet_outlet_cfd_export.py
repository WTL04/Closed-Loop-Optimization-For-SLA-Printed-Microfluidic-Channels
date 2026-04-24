import cadquery as cq
import subprocess
import os
import sys

# Fixed nominal dimensions (mm)
BASE_LENGTH = 40.0
BASE_WIDTH = 0.5
BASE_HEIGHT = 0.5

# CFD extensions (mm)
INLET_LENGTH = 5.0
OUTLET_LENGTH = 5.0

# Shift to keep geometry inside blockMesh background mesh
X_SHIFT = 30.0

# Paths
STL_EXPORT_DIR = "/home/will/Downloads/coding/uni/ml-research/contextual_bayes_opt/cfd/channelCase/constant/triSurface/"
CFD_RUN_SCRIPT = (
    "/home/will/Downloads/coding/uni/ml-research/contextual_bayes_opt/run_cfd.sh"
)
SHEET_NAME = "Experiment Realistic Deltas"


def compute_post_print_dimensions(length_delta, width_delta, height_delta):
    """
    Deltas are shrinkage values in micrometres (µm).
    Actual printed dimension = nominal - shrinkage.
    Convert µm -> mm by dividing by 1000.
    """
    L = BASE_LENGTH - (length_delta / 1000.0)
    W = BASE_WIDTH - (width_delta / 1000.0)
    H = BASE_HEIGHT - (height_delta / 1000.0)

    if L <= 0 or W <= 0 or H <= 0:
        raise ValueError(
            f"Post-print dimensions became non-positive: L={L}, W={W}, H={H}. "
            f"Check delta units — expected micrometres."
        )
    return L, W, H


def build_channel(length_delta, width_delta, height_delta):
    L, W, H = compute_post_print_dimensions(length_delta, width_delta, height_delta)
    total_length = INLET_LENGTH + L + OUTLET_LENGTH

    # World-space X coordinates of channel ends
    x_start = X_SHIFT  # left end (inlet face)
    x_end = X_SHIFT + total_length  # right end (outlet face)
    x_mid = X_SHIFT + total_length / 2.0  # centre for box placement
    y_mid = 0.0
    z_mid = H / 2.0

    # --- Fluid volume: full enclosed rectangular prism ---
    fluid = cq.Workplane("XY").box(total_length, W, H).translate((x_mid, y_mid, z_mid))

    # --- Inlet: flat face at x_start (near-zero thickness for STL export) ---
    # Uses YZ plane so the face normal points along X
    inlet = (
        cq.Workplane("YZ")
        .rect(W, H)
        .extrude(0.001)  # 1 µm thickness — effectively a surface
        .translate((x_start, y_mid, z_mid))
    )

    # --- Outlet: flat face at x_end ---
    outlet = (
        cq.Workplane("YZ")
        .rect(W, H)
        .extrude(0.001)
        .translate(
            (x_end - 0.001, y_mid, z_mid)
        )  # offset back by thickness so it sits flush
    )

    # --- Walls: fluid box surface minus the two end faces ---
    # Select all faces except the -X face (inlet end) and +X face (outlet end)
    # These are the 4 long faces: top, bottom, left, right
    walls = (
        cq.Workplane("XY")
        .box(total_length, W, H)
        .translate((x_mid, y_mid, z_mid))
        .faces("not <X and not >X")  # exclude the two end faces
        .shell(0.001)  # near-zero shell to make exportable surface
    )

    return fluid, walls, inlet, outlet, L, W, H


def run_pipeline(length_delta, width_delta, height_delta):
    fluid, walls, inlet, outlet, L, W, H = build_channel(
        length_delta, width_delta, height_delta
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

    try:
        subprocess.run(["bash", CFD_RUN_SCRIPT], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"CFD simulation failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 3:
        l_d = float(sys.argv[1])
        w_d = float(sys.argv[2])
        h_d = float(sys.argv[3])
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
            l_d = float(
                sheets_api.get_latest_col_value(cols["length"], sheet_name=SHEET_NAME)
                or 0.0
            )
            w_d = float(
                sheets_api.get_latest_col_value(cols["width"], sheet_name=SHEET_NAME)
                or 0.0
            )
            h_d = float(
                sheets_api.get_latest_col_value(cols["height"], sheet_name=SHEET_NAME)
                or 0.0
            )
        except Exception as e:
            print(f"Could not fetch from sheets: {e}. Using zero deltas.")
            l_d, w_d, h_d = 0.0, 0.0, 0.0

    success = run_pipeline(l_d, w_d, h_d)
    sys.exit(0 if success else 1)

