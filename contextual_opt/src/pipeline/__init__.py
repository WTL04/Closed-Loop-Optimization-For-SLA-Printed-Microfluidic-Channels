"""
Contextual Bayesian Optimization Pipeline

Submodules:
- search_space: build_search_space
- context: get_context_snapshot, context_overtime
- runner: run_single_channel
- utils: print_suggested_params, save_params_to_json, append_single_to_sheets
- data_loader: load_dataset, load_data_source, extract_channel_data
- delta_loaders: generate_random_deltas, generate_realistic_deltas
- metrics: compute_dimensional_error, compute_flow_rate_cv
- cfd_runs: run_cfd_simulation
- cad_model: build_cad_model, export_cad_model
- run: Main orchestration (run_with_google_sheets, run_with_testing, run_sequential)
- config: Configuration constants
"""

from .config import *

from .metrics import (
    compute_flow_rate_cv,
    compute_dimensional_error,
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

from .cfd_runs import (
    run_cfd_simulation,
)

from .cad_model import (
    build_cad_model,
    export_cad_model,
)

from .search_space import build_search_space

from .context import get_context_snapshot, context_overtime

from .runner import run_single_channel

from .utils import print_suggested_params, save_params_to_json, append_single_to_sheets

from .run import (
    run_with_google_sheets,
    run_with_testing,
    run_sequential,
)
