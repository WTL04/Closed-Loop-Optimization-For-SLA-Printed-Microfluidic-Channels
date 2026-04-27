# Automated CAD to CFD Pipeline

## Overview

This guide explains how to set up the Python environment and execute the automated OpenFOAM CFD pipeline for microfluidic channel optimization.

This assumes:
* You have cloned this repository.
* Docker is installed and running on your system (required for containerized OpenFOAM execution).

---

# 1. Environment Setup

Set up a virtual environment to install the required Python dependencies (including `pandas`, `gspread`, and CadQuery libraries). You can use either standard Python `venv` or `conda`.

**Option A: Using venv**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r req.txt
```

**Option B: Using Conda**
```bash
conda create -n cfd_env python=3.11 -y
conda activate cfd_env
pip install -r req.txt
```

---

# 2. Run a Single Channel Test

To test the CFD pipeline on a single geometry configuration, execute the bash execution script. This script automatically handles CAD generation, meshing, solving, and flow rate extraction.

**Test with specific dimensional deltas (in micrometers):**
Pass the length, width, and height deltas as positional arguments.
```bash
./run_cfd.sh <length_delta> <width_delta> <height_delta>
```
*Example:* `./run_cfd.sh 10.0 5.0 -2.5`

**Test the nominal geometry (0 delta):**
If no parameters are provided, the script defaults to simulating the base channel dimensions.
```bash
./run_cfd.sh
```

Upon completion, the extracted flow rate will be saved to `cfd/channelCase/flow_rate.txt` and printed directly to the terminal.

---

# 3. Full Dataset Automation

To process an entire experimental batch and sync the computational results with your database, run the master controller:

```bash
python pipeline_master.py
```

**What this does:**
1. Asks user which page in sheets to read/write into.
2. Fetches pending experimental parameters from the connected Google Sheet.
3. Iterates through all channels in each batch, applying the specific post-print dimensional deltas.
4. Triggers `./run_cfd.sh` to simulate each configuration.
5. Automatically writes the calculated `mL/min` flow rates and the Batch Coefficient of Variation (CV) back into the respective cells in the Google Sheet.


*Troubleshooting:* If a specific channel fails during the automated loop, check the terminal output for `checkMesh` errors (output from openFOAM). Failures here typically indicate that the CBO or dataset provided a physically impossible dimensional delta (e.g., negative cell volumes due to a geometry that exceeds the bounding box). The pipeline will safely log this failure and continue to the next channel.
