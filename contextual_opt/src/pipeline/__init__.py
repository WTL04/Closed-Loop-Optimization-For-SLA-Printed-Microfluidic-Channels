"""
Contextual Bayesian Optimization Pipeline

Submodules:
- computation: compute_flow_rate_cv, compute_dimensional_error, calculate_functional_recovery
- data_loader: load_dataset, load_data_source, extract_channel_data
- delta_loaders: generate_random_deltas, generate_realistic_deltas
- trial_runs: run_cfd_simulation, run_fake_trial, run_real_trial
- cad_model: build_cad_model, export_cad_model
- run: Main orchestration (run.py)
- config: Configuration constants
"""
from .config import *

from .computation import (
    compute_flow_rate_cv,
    compute_dimensional_error,
    calculate_functional_recovery,
)

from .data_loader import (
    load_dataset,
    load_data_source,
    extract_channel_data,
)

from .delta_loaders import (
    generate_random_deltas,
    generate_realistic_deltas,
)

from .trial_runs import (
    run_cfd_simulation,
    simulate_print_trial,
    run_fake_trial,
    run_real_trial,
)

from .cad_model import (
    build_cad_model,
    export_cad_model,
)