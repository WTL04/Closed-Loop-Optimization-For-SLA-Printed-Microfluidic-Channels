# Configuration constants
NUM_CHANNELS = 4
MIN_CHANNEL_SPACING = 2.0  # mm - minimum gap between adjacent channel rows

# Channel dimension bounds (same for all channels)
# TODO: Adjust to real range
CHANNEL_LENGTH_BOUNDS = (10.0, 18.0)  # mm
CHANNEL_WIDTH_BOUNDS = (0.1, 0.5)  # mm
CHANNEL_HEIGHT_BOUNDS = (0.1, 0.7)  # mm

# Base plate dimensions for CAD model
# TODO: Adjust to real dimensions
BASE_LENGTH = 20.0  # mm (X direction)
BASE_WIDTH = 10.0  # mm (Y direction - channels arranged along this axis)
BASE_THICKNESS = 1.0  # mm
