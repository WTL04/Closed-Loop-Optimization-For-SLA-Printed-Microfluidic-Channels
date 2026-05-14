"""
Trial execution functions for CBO and CFD.

Functions:
- run_cfd_simulation
"""

import subprocess
from contextual_opt.src.pipeline.config import CFD_RUN_SCRIPT


def run_cfd_simulation(
    cbo_length_um: float,
    cbo_width_um: float,
    cbo_height_um: float,
    length_delta: float,
    width_delta: float,
    height_delta: float,
) -> float:
    """
    Run the full CFD pipeline by calling run_cfd.sh with CBO CAD inputs and deltas.
    Expected Physical = CBO_Suggested_CAD + Printer_Delta_Error
    Blocks until the simulation completes and returns the extracted flow rate.

    Args:
        cbo_length_um: CBO-suggested CAD length in µm
        cbo_width_um:  CBO-suggested CAD width in µm
        cbo_height_um: CBO-suggested CAD height in µm
        length_delta:  Post-print length shrinkage in µm
        width_delta:   Post-print width shrinkage in µm
        height_delta:  Post-print height shrinkage in µm

    Returns:
        Flow rate in m^3/s extracted from OpenFOAM
    """
    try:
        print(
            f"Running CFD: CBO inputs L={cbo_length_um:.2f} W={cbo_width_um:.2f} H={cbo_height_um:.2f} µm, "
            f"deltas L={length_delta:.2f} W={width_delta:.2f} H={height_delta:.2f} µm"
        )
        print("Please wait (may take a few minutes)...")
        result = subprocess.run(
            [
                "bash",
                CFD_RUN_SCRIPT,
                str(cbo_length_um),
                str(cbo_width_um),
                str(cbo_height_um),
                str(length_delta),
                str(width_delta),
                str(height_delta),
            ],
            check=True,
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
