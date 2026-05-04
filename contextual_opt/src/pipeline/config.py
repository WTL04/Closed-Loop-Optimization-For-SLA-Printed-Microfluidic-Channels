# Configuration constants
NUM_CHANNELS = 1
MIN_CHANNEL_SPACING = 2.0  # mm - minimum gap between adjacent channel rows

# Baseline ground truth for functional validation
BASELINE_FLOW_RATE = 2.312215e-9  # m³/s - measured using openFOAM with no 0.0 deltas

# Channel dimension bounds (pre-distortion limits)
CHANNEL_LENGTH_BOUNDS = (40.0, 40.05)  # mm
CHANNEL_WIDTH_BOUNDS = (0.50, 0.52)  # mm
CHANNEL_HEIGHT_BOUNDS = (0.50, 0.52)  # mm

# Base plate dimensions for CAD model
# TODO: Adjust to real dimensions
BASE_LENGTH = 20.0  # mm (X direction)
BASE_WIDTH = 10.0  # mm (Y direction - channels arranged along this axis)
BASE_THICKNESS = 1.0  # mm
