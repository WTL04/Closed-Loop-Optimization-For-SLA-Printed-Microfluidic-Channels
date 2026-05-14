from pathlib import Path

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
        if not dat_file.exists():
            dat_file = latest / "surfaceFieldValue.dat"
    except (ValueError, IndexError, OSError):
        print("No timestep directories found")
        return 0.0

    if not dat_file.exists():
        print("No surfaceFieldValue dat file found")
        return 0.0

    try:
        with open(dat_file, "r") as f:
            # Filter out empty lines and comment lines
            valid_lines = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        if not valid_lines:
            print("Dat file contains no valid data.")
            return 0.0

        # The last line should contain [Time, FlowRate]
        last_line_data = valid_lines[-1].split()

        # Ensure the line actually has at least two columns before extracting
        if len(last_line_data) >= 2:
            flow_rate = float(last_line_data[1])
            return flow_rate
        else:
            print("Last line was incomplete.")
            return 0.0

    except Exception as e:
        print(f"Error reading dat file: {e}")
        return 0.0


if __name__ == "__main__":
    flow_rate = extract_flow_rate()
    print(f"FLOW_RATE:{flow_rate}")
