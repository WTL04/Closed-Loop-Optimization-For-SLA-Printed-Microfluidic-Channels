# Configuration constants
NUM_CHANNELS = 4
MIN_CHANNEL_SPACING = 2.0  # mm - minimum gap between adjacent channel rows

# Baseline ground truth for functional validation
BASELINE_FLOW_RATE = 2.312215e-9  # m³/s - measured using openFOAM with no 0.0 deltas

# Channel dimension bounds in micrometers (pre-distortion: nominal to nominal + max expected delta)
CHANNEL_LENGTH_UM_BOUNDS = (40000, 40060)  # µm (nominal + 60µm max delta)
CHANNEL_WIDTH_UM_BOUNDS = (500, 515)       # µm (nominal + 15µm max delta)
CHANNEL_HEIGHT_UM_BOUNDS = (500, 515)      # µm (nominal + 15µm max delta)

# Nominal target dimensions in micrometers (for error computation)
NOMINAL_DIMENSIONS = {
    "length": 40000,
    "width": 500,
    "height": 500,
}

# Base plate dimensions for CAD model
BASE_LENGTH = 20.0  # mm (X direction)
BASE_WIDTH = 10.0  # mm (Y direction - channels arranged along this axis)
BASE_THICKNESS = 1.0  # mm

# Path to the shell script that runs the full CFD pipeline
CFD_RUN_SCRIPT = "run_cfd.sh"
