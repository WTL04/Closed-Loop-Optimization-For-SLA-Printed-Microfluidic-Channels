from pathlib import Path
import numpy as np

CASE_DIR = Path("cfd/channelCase")
POST_DIR = CASE_DIR / "postProcessing" / "flowRatePatch(name=outlet)"


def extract_flow_rate() -> float:
    """Extract flow rate from OpenFOAM v1912 function object output."""
    if not POST_DIR.exists():
        print("postProcessing/flowRatePatch(name=outlet) not found")
        return 0.0

    try:
        latest = sorted(POST_DIR.iterdir(), key=lambda p: float(p.name))[-1]
        dat_file = latest / "surfaceFieldValue_0.dat"
    except (ValueError, IndexError, OSError):
        print("No timestep directories found")
        return 0.0

    if not dat_file.exists():
        print("surfaceFieldValue_0.dat not found")
        return 0.0

    try:
        dat = np.loadtxt(dat_file, comments="#")
        flow_rate = float(dat[-1, 1])
        return flow_rate
    except Exception as e:
        print(f"Error reading dat file: {e}")
        return 0.0


if __name__ == "__main__":
    flow_rate = extract_flow_rate()
    print(f"FLOW_RATE:{flow_rate}")

