# Configuration constants
NUM_CHANNELS = 4
MIN_CHANNEL_SPACING = 2.0  # mm - minimum gap between adjacent channel rows

# Baseline ground truth for functional validation
BASELINE_FLOW_RATE = 2.312215e-9  # m³/s - measured using openFOAM with no 0.0 deltas

# Channel dimension bounds (pre-distortion limits)
CHANNEL_LENGTH_BOUNDS = (39.95, 40.05)  # mm (±0.05mm)
CHANNEL_WIDTH_BOUNDS = (0.45, 0.55)  # mm (±0.05mm)
CHANNEL_HEIGHT_BOUNDS = (0.45, 0.55)  # mm (±0.05mm)

# Nominal target dimensions (for error computation)
NOMINAL_DIMENSIONS = {
    "length": 40.0,
    "width": 0.5,
    "height": 0.5,
}

# Base plate dimensions for CAD model
BASE_LENGTH = 20.0  # mm (X direction)
BASE_WIDTH = 10.0  # mm (Y direction - channels arranged along this axis)
BASE_THICKNESS = 1.0  # mm

# Path to the shell script that runs the full CFD pipeline
CFD_RUN_SCRIPT = "run_cfd.sh"
