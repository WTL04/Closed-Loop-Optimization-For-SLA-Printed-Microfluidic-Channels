"""
Trial execution functions for CBO and CFD.

Functions:
- run_cfd_simulation
"""

import subprocess
from contextual_opt.src.pipeline.config import CFD_RUN_SCRIPT


def run_cfd_simulation(
    length_delta: float, width_delta: float, height_delta: float
) -> float:
    """
    Run the full CFD pipeline by calling run_cfd.sh with delta arguments.
    Blocks until the simulation completes and returns the extracted flow rate.

    Args:
        length_delta: Post-print length shrinkage in µm
        width_delta:  Post-print width shrinkage in µm
        height_delta: Post-print height shrinkage in µm

    Returns:
        Flow rate in m^3/s extracted from OpenFOAM
    """
    try:
        print(
            f"Running CFD: deltas L={length_delta:.2f} W={width_delta:.2f} H={height_delta:.2f} µm"
        )
        print("Please wait (may take a few minutes)...")
        subprocess.run(
            [
                "bash",
                CFD_RUN_SCRIPT,
                str(length_delta),
                str(width_delta),
                str(height_delta),
            ],
            check=True,
            capture_output=True,
        )

        # Read extracted flow rate
        with open("cfd/channelCase/flow_rate.txt", "r") as f:
            content = f.read().strip()
            if "FLOW_RATE:" in content:
                return float(content.split("FLOW_RATE:")[1])
            return 0.0

    except Exception as e:
        print(f"CFD simulation failed: {e}")
        return 0.0
