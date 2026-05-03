import subprocess
import os
import numpy as np
import pandas as pd

# Import your existing API script
from contextual_opt.src.api import sheets_api

# Configuration
SHEET_NAME = "Experiment Random Deltas"
FLOW_RATE_FILE = "cfd/channelCase/flow_rate.txt"
CONVERSION_FACTOR = 1e6 * 60  # OpenFOAM m^3/s to mL/min


def run_full_automation():
    SHEET_NAME = input("Input sheet name: ")

    # pull full sheet, return as Pandas DataFrame
    df = sheets_api.pullData(sheet_name=SHEET_NAME, verbose=False)

    if df is None:
        print("Sheet not found, please input valid sheet name")
        return

    print(f"Fetching data from Google Sheets: {SHEET_NAME}...")

    for index, row in df.iterrows():
        batch_id = row["batch_id"]
        print(f"\n========================================")
        print(f"Processing Batch {batch_id}")
        print(f"========================================")

        flow_rates = []
        updates = {}

        for ch in range(1, 5):
            # Extract deltas. If the sheet is empty (NaN), default to 0.0 for nominal geometry
            l_delta = row.get(f"channel_{ch}_post_print_length_delta")
            w_delta = row.get(f"channel_{ch}_post_print_width_delta")
            h_delta = row.get(f"channel_{ch}_post_print_height_delta")

            l_delta = 0.0 if pd.isna(l_delta) or l_delta == "" else float(l_delta)
            w_delta = 0.0 if pd.isna(w_delta) or w_delta == "" else float(w_delta)
            h_delta = 0.0 if pd.isna(h_delta) or h_delta == "" else float(h_delta)

            print(
                f"  [Channel {ch}] Deltas (um): L={l_delta}, W={w_delta}, H={h_delta}"
            )

            # Execute OpenFOAM pipeline
            result = subprocess.run(
                ["./run_cfd.sh", str(l_delta), str(w_delta), str(h_delta)],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"  [!] CFD Failed for Channel {ch}. Check logs.")
                continue

            # Extract flow rate via flow_rate.txt, convert cubic meters per second to millileter per minute
            try:
                with open(FLOW_RATE_FILE, "r") as f:
                    content = f.read().strip()
                    if "FLOW_RATE:" in content:
                        raw_val_m3s = float(content.split("FLOW_RATE:")[1])
                        val_ml_min = raw_val_m3s * CONVERSION_FACTOR

                        flow_rates.append(val_ml_min)
                        updates[f"channel_{ch}_flow_rate_ml_per_min"] = val_ml_min
                        print(f"  -> Success: {val_ml_min:.6f} mL/min")
                    else:
                        print(f"  [!] Unrecognized output in flow_rate.txt")
            except FileNotFoundError:
                print(f"  [!] Error: {FLOW_RATE_FILE} missing.")

        # Calculate Coefficient of Variation (CV) across the 4 channels
        if len(flow_rates) > 1:
            mean_fr = np.mean(flow_rates)
            if mean_fr != 0:
                cv = np.std(flow_rates) / mean_fr
                updates["flow_rate_cv"] = cv
                print(f"  -> Batch CV Calculated: {cv:.4f}")

        # Push the row's simulation results to Google Sheets
        if updates:
            print(f"  Pushing updates to Google Sheets for Batch {batch_id}...")
            try:
                sheets_api.update_row(
                    batch_id=batch_id, updates=updates, sheet_name=SHEET_NAME
                )
                print(f"  -> Update confirmed.")
            except Exception as e:
                print(f"  [!] Google Sheets update failed: {e}")


if __name__ == "__main__":
    run_full_automation()
