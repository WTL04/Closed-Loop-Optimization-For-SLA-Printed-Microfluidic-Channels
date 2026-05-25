"""
Delta generation functions for simulating post-print shrinkage.

Functions:
- generate_random_deltas
- generate_realistic_deltas
"""

import numpy as np


def generate_random_deltas() -> dict:
    """
    Generate random post-print deltas using uniform distribution (0–19 µm).
    Serves as the stationary baseline condition with no context coupling.

    Returns:
        dict with length, width, height delta values in µm
    """
    return {
        "length": np.random.uniform(0, 19),
        "width": np.random.uniform(0, 19),
        "height": np.random.uniform(0, 19),
    }


def generate_realistic_deltas(
    ambient_temp: float = 75.0,
    resin_age_hours: float = 0.0,
    nominal_temp: float = 75.0,
) -> dict:
    """
    Generate realistic post-print deltas using a three-component distribution:
    - Gaussian core (normal process variation)
    - Positive skew term (overcuring effects)
    - Stochastic outliers (fabrication anomalies)

    Context variables modulate the distribution parameters rather than
    adding arbitrary penalties. This preserves the original methodology
    while coupling shrinkage to the physical environment.

    Args:
        ambient_temp:     Current ambient temperature (°F)
        resin_age_hours:  Current resin age (hours since opened)
        nominal_temp:     Baseline reference temperature (°F), default 75°F

    Returns:
        dict with length, width, height delta values in µm


    NOTE FOR FUTURE WORK: The scalar coefficients governing the context-to-delta coupling
    (temp_deviation * 0.08, resin_age_hours * 0.01, etc.) are simulation
    design parameters, not empirically measured physical constants. Their
    values were selected to produce a non-stationary shrinkage distribution
    consistent with the qualitative mechanisms described in Pan et al. (2017)
    and Kim et al. (2025), while constraining delta output to the physically
    plausible range of 0–20 µm for SLA fabrication.

    For in-situ validation, replace these coefficients with values
    empirically fitted from physical print measurements at controlled
    temperature and resin age conditions.
    """
    temp_deviation = ambient_temp - nominal_temp  # positive = hot, negative = cold

    # --- Gaussian core ---
    # Hot: higher mean (overcuring expands cured volume, increasing shrinkage on cool)
    # Cold: higher std (viscosity variation makes errors less predictable)
    # Age: gradually raises mean as resin accumulates exothermic heat history
    base_mean = 8.0 + (temp_deviation * 0.08) + (resin_age_hours * 0.01)
    base_std = 2.5 + (abs(temp_deviation) * 0.04)

    # Clamp mean to physically plausible range (2–15 µm)
    base_mean = np.clip(base_mean, 2.0, 15.0)
    base_std = np.clip(base_std, 1.0, 5.0)

    deltas = {}
    for dim in ["length", "width", "height"]:
        core = np.random.normal(base_mean, base_std)

        # --- Overcure skew ---
        # Probability and magnitude increase with heat
        overcure_prob = np.clip(0.3 + (max(0, temp_deviation) * 0.005), 0.3, 0.55)
        overcure = 0.0
        if np.random.random() < overcure_prob:
            overcure = np.random.uniform(0, 5)

        # --- Outliers ---
        # Probability increases slightly with age (resin degradation)
        outlier_prob = np.clip(0.05 + (resin_age_hours * 0.0003), 0.05, 0.12)
        outlier = 0.0
        if np.random.random() < outlier_prob:
            outlier = np.random.uniform(8, 15)

        total_delta = core + overcure + outlier
        deltas[dim] = np.clip(total_delta, 0.0, 23.0)

    return deltas
