from pathlib import Path

CASE_DIR = Path("cfd/channelCase")
POST_DIR = CASE_DIR / "postProcessing" / "flowRatePatch(name=outlet)"


def extract_flow_rate() -> float:
    """Extract flow rate from OpenFOAM v1912 function object output.

    Returns:
        Flow rate in m³/s, or -1.0 if extraction failed.
    """
    if not POST_DIR.exists():
        print(
            "ERROR: postProcessing/flowRatePatch(name=outlet) not found - simulation likely diverged"
        )
        return -1.0

    try:
        latest = sorted(POST_DIR.iterdir(), key=lambda p: float(p.name))[-1]
        dat_file = latest / "surfaceFieldValue_0.dat"
        if not dat_file.exists():
            dat_file = latest / "surfaceFieldValue.dat"
    except (ValueError, IndexError, OSError):
        print("ERROR: No timestep directories found")
        return -1.0

    if not dat_file.exists():
        print("ERROR: No surfaceFieldValue dat file found")
        return -1.0

    try:
        with open(dat_file, "r") as f:
            valid_lines = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        if not valid_lines:
            print("ERROR: Dat file contains no valid data")
            return -1.0

        last_line_data = valid_lines[-1].split()

        if len(last_line_data) >= 2:
            flow_rate = float(last_line_data[1])
            if flow_rate <= 0:
                print(
                    f"WARNING: Non-positive flow rate {flow_rate}, marking as failure"
                )
                return -1.0
            return flow_rate
        else:
            print("ERROR: Last line was incomplete")
            return -1.0

    except Exception as e:
        print(f"ERROR: Exception reading dat file: {e}")
        return -1.0


if __name__ == "__main__":
    flow_rate = extract_flow_rate()
    print(f"FLOW_RATE:{flow_rate}")
