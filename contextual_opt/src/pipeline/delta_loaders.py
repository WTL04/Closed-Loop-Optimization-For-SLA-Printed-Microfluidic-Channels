"""
Delta generation functions for simulating post-print shrinkage.

Functions:
- generate_random_deltas
- generate_realistic_deltas
"""

import numpy as np


def generate_random_deltas(mode: str = "uniform") -> dict:
    """
    Generate random post-print deltas using uniform distribution.

    Args:
        mode: Distribution type (currently only uniform supported)

    Returns:
        dict with length, width, height delta values in µm
    """
    return {
        "length": np.random.uniform(0, 20),
        "width": np.random.uniform(0, 20),
        "height": np.random.uniform(0, 20),
    }


def generate_realistic_deltas() -> dict:
    """
    Generate realistic post-print deltas based on empirical distribution.

    Based on observed data patterns from experiments.
    Typical range: 5-15 µm for length, 5-20 µm for width/height.

    Returns:
        dict with length, width, height delta values in µm
    """
    return {
        "length": np.random.normal(10, 5),
        "width": np.random.normal(12, 7),
        "height": np.random.normal(10, 6),
    }

