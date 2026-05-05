"""
Search space building for Ax CBO.

Functions:
- build_search_space: Create Ax SearchSpace for single channel optimization
"""

from ax.core import (
    SearchSpace,
    RangeParameter,
    ParameterType,
)
from contextual_opt.src.pipeline.config import (
    CHANNEL_LENGTH_UM_BOUNDS,
    CHANNEL_WIDTH_UM_BOUNDS,
    CHANNEL_HEIGHT_UM_BOUNDS,
)
from contextual_opt.src.pipeline.config import (
    AMBIENT_TEMP_BOUNDS,
    RESIN_TEMP_BOUNDS,
    RESIN_AGE_BOUNDS,
)


def build_search_space(num_channels: int = 1):
    """
    Build search space for SINGLE channel optimization.

    Args:
        num_channels: Number of channels (default 1 for decoupled approach)

    Returns:
        SearchSpace with:
        - Knobs: channel_length_um, channel_width_um, channel_height_um
        - Context params (registered for indexing): layer_thickness_um, ambient_temp, resin_temp, resin_age
    """
    parameters = []

    # Geometric knobs - pre-distortion targets to optimize (in µm)
    parameters.extend(
        [
            RangeParameter(
                name="channel_length_um",
                parameter_type=ParameterType.FLOAT,
                lower=CHANNEL_LENGTH_UM_BOUNDS[0],
                upper=CHANNEL_LENGTH_UM_BOUNDS[1],
            ),
            RangeParameter(
                name="channel_width_um",
                parameter_type=ParameterType.FLOAT,
                lower=CHANNEL_WIDTH_UM_BOUNDS[0],
                upper=CHANNEL_WIDTH_UM_BOUNDS[1],
            ),
            RangeParameter(
                name="channel_height_um",
                parameter_type=ParameterType.FLOAT,
                lower=CHANNEL_HEIGHT_UM_BOUNDS[0],
                upper=CHANNEL_HEIGHT_UM_BOUNDS[1],
            ),
        ]
    )

    # Context parameters - registered in search space for surrogate indexing
    # Passed via c_t in suggest(), values fixed at suggestion time
    parameters.extend(
        [
            RangeParameter(
                name="layer_thickness_um",
                parameter_type=ParameterType.FLOAT,
                lower=50.0,
                upper=100.0,
            ),
            RangeParameter(
                name="ambient_temp",
                parameter_type=ParameterType.FLOAT,
                lower=AMBIENT_TEMP_BOUNDS[0],
                upper=AMBIENT_TEMP_BOUNDS[1],
            ),
            RangeParameter(
                name="resin_temp",
                parameter_type=ParameterType.FLOAT,
                lower=RESIN_TEMP_BOUNDS[0],
                upper=RESIN_TEMP_BOUNDS[1],
            ),
            RangeParameter(
                name="resin_age",
                parameter_type=ParameterType.FLOAT,
                lower=RESIN_AGE_BOUNDS[0],
                upper=RESIN_AGE_BOUNDS[1],
            ),
        ]
    )

    # NOTE: delta_*_um are outputs (manufacturing errors), not knobs
    # They are measured after fabrication and used in dim_error calculation

    return SearchSpace(parameters=parameters)

