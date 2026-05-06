# Configuration constants
NUM_CHANNELS = 4
MIN_CHANNEL_SPACING = 2.0  # mm - minimum gap between adjacent channel rows

# Baseline ground truth for functional validation
BASELINE_FLOW_RATE = 2.312215e-9  # m³/s - measured using openFOAM with no 0.0 deltas

# Channel dimension bounds in micrometers (pre-distortion: nominal +/- max expected delta)
CHANNEL_LENGTH_UM_BOUNDS = (39940, 40060)
CHANNEL_WIDTH_UM_BOUNDS = (485, 515)
CHANNEL_HEIGHT_UM_BOUNDS = (485, 515)

# Nominal target dimensions in micrometers (for error computation)
NOMINAL_DIMENSIONS = {
    "length": 40000,
    "width": 500,
    "height": 500,
}

# Temperature bounds in Fahrenheit
AMBIENT_TEMP_BOUNDS = (60.0, 100.0)  # F
RESIN_TEMP_BOUNDS = (60.0, 100.0)  # F

# Resin age bounds in hours
RESIN_AGE_BOUNDS = (0.0, 72.0)  # hours

# Context overtime drift settings (in Fahrenheit)
CONTEXT_DRIFT_MAX = 3.0  # max drift per channel trial
CONTEXT_NOISE_MAX = 2.0  # max random noise

# Base plate dimensions for CAD model
BASE_LENGTH = 20.0  # mm (X direction)
BASE_WIDTH = 10.0  # mm (Y direction - channels arranged along this axis)
BASE_THICKNESS = 1.0  # mm

# Path to the shell script that runs the full CFD pipeline
CFD_RUN_SCRIPT = "run_cfd.sh"
