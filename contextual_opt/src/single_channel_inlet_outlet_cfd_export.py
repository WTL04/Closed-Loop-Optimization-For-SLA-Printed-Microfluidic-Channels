import cadquery as cq
import subprocess
import os
import sys
from contextual_opt.src import sheets_api

# Fixed nominal dimensions (mm)
BASE_LENGTH = 40.0
BASE_WIDTH = 0.5
BASE_HEIGHT = 0.5

# CFD extensions (mm)
INLET_LENGTH = 5.0
OUTLET_LENGTH = 5.0

# Shift to keep geometry inside mesh
X_SHIFT = 30.0

# Paths
STL_EXPORT_DIR = "/home/will/Downloads/coding/uni/ml-research/contextual_bayes_opt/cfd/channelCase/constant/triSurface/"
CFD_RUN_SCRIPT = (
    "/home/will/Downloads/coding/uni/ml-research/contextual_bayes_opt/run_cfd.sh"
)
SHEET_NAME = "Experiment Realistic Deltas"


def compute_post_print_dimensions(length_delta, width_delta, height_delta):
    L = BASE_LENGTH + length_delta
    W = BASE_WIDTH + width_delta
    H = BASE_HEIGHT + height_delta

    if L <= 0 or W <= 0 or H <= 0:
        raise ValueError("Post-print dimensions became non-positive!")

    return L, W, H


def build_channel(length_delta, width_delta, height_delta):
    L, W, H = compute_post_print_dimensions(length_delta, width_delta, height_delta)

    total_length = INLET_LENGTH + L + OUTLET_LENGTH

    fluid = (
        cq.Workplane("XY").box(total_length, W, H).translate((X_SHIFT, 0.0, H / 2.0))
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


def run_pipeline(length_delta, width_delta, height_delta):
    fluid, walls, inlet, outlet, L, W, H = build_channel(
        length_delta, width_delta, height_delta
    )

    os.makedirs(STL_EXPORT_DIR, exist_ok=True)

    cq.exporters.export(fluid, os.path.join(STL_EXPORT_DIR, "channels_fluid.stl"))
    cq.exporters.export(walls, os.path.join(STL_EXPORT_DIR, "channel_walls.stl"))
    cq.exporters.export(inlet, os.path.join(STL_EXPORT_DIR, "channel_inlet.stl"))
    cq.exporters.export(outlet, os.path.join(STL_EXPORT_DIR, "channel_outlet.stl"))

    print(f"Generated STL files for L={L}, W={W}, H={H}")

    try:
        subprocess.run(["bash", CFD_RUN_SCRIPT], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"CFD simulation failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 3:
        # Run with provided deltas
        l_d = float(sys.argv[1])
        w_d = float(sys.argv[2])
        h_d = float(sys.argv[3])
    else:
        # Fallback to sheets latest
        print(f"No arguments provided. Fetching latest from {SHEET_NAME}...")
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
            sheets_api.get_latest_col_value(cols["width"], sheet_name=SHEET_NAME) or 0.0
        )
        h_d = float(
            sheets_api.get_latest_col_value(cols["height"], sheet_name=SHEET_NAME)
            or 0.0
        )

    success = run_pipeline(l_d, w_d, h_d)
    sys.exit(0 if success else 1)
