import json
import numpy as np
from pathlib import Path

from ..api.sheets_api import update_row

NUM_CHANNELS = 4


def compute_flow_rate_cv(flow_values: list) -> float:
    if len(flow_values) < 2:
        return 0.0
    return float(np.std(flow_values) / np.mean(flow_values))


def load_params_from_json(filename: str = "suggested_params.json") -> dict:
    # Look in contextual_opt/src/data/, fallback to current directory
    search_dirs = [
        Path(__file__).parent.parent / "data",
        Path.cwd(),
    ]
    for search_dir in search_dirs:
        filepath = search_dir / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"Could not find {filename} in {search_dirs}")


if __name__ == "__main__":
    suggested_params = load_params_from_json()
    batch_id = int(suggested_params.get("batch_id", 1))

    print("\nEnter post-print measurements for each channel:")
    warp_deltas = {}
    for i in range(1, NUM_CHANNELS + 1):
        print(f"  Channel {i}:")
        for dim in ("length", "width", "height"):
            intended = suggested_params[f"channel_{i}_{dim}"]
            measured = float(
                input(f"    Measured {dim} (intended={intended:.3f} mm): ")
            )
            warp_deltas[f"channel_{i}_post_print_{dim}_delta"] = measured - intended

    print("\nEnter measured flow rates for each channel:")
    flow_rates = {}
    flow_values = []
    for i in range(1, NUM_CHANNELS + 1):
        val = float(input(f"  channel_{i}_flow_rate_ml_per_min: "))
        flow_rates[f"channel_{i}_flow_rate_ml_per_min"] = val
        flow_values.append(val)

    cv = compute_flow_rate_cv(flow_values)
    flow_rates["flow_rate_cv"] = cv
    print(f"  Computed flow_rate_cv: {cv:.6f}")

    update_row(batch_id, warp_deltas, sheet_name="Ax")
    update_row(batch_id, flow_rates, sheet_name="Ax")

    print(f"\nUpdated batch {batch_id} in Google Sheets with measurements.")
    print("Run ax_run_cbo.py again to get next suggestion.")
