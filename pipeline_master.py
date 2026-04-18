import os
import subprocess
import pandas as pd
from contextual_opt.src import sheets_api
from contextual_opt.src.single_channel_inlet_outlet_cfd_export import run_pipeline

# Configuration
SHEET_NAME = "Experiment Realistic Deltas"
FLOW_RATE_FILE = "/home/will/Downloads/coding/uni/ml-research/contextual_bayes_opt/cfd/channelCase/flow_rate.txt"
# Note: run_cfd.sh might output flow_rate.txt in cfd/channelCase
# Let's make sure we know the exact path.

def extract_flow_rate_from_file():
    if not os.path.exists(FLOW_RATE_FILE):
        return None
    with open(FLOW_RATE_FILE, "r") as f:
        for line in f:
            if "FLOW_RATE:" in line:
                return float(line.split(":")[1].strip())
    return None

def main():
    print(f"Starting master pipeline for sheet: {SHEET_NAME}")
    df = sheets_api.pullData(sheet_name=SHEET_NAME, verbose=False)
    
    # Identify the columns for the first channel to iterate over
    # We assume we want to run for all rows that have deltas
    # For simplicity, we'll iterate over all rows in the sheet
    
    for index, row in df.iterrows():
        batch_id = row.get("batch_id")
        if batch_id is None:
            continue
            
        print(f"\nProcessing Batch ID: {batch_id}")
        
        try:
            l_d = float(row.get("channel_1_post_print_length_delta", 0.0))
            w_d = float(row.get("channel_1_post_print_width_delta", 0.0))
            h_d = float(row.get("channel_1_post_print_height_delta", 0.0))
            
            print(f"Deltas: L={l_d}, W={w_d}, H={h_d}")
            
            # 1 & 2: Export STL and Run Simulation
            success = run_pipeline(l_d, w_d, h_d)
            
            if success:
                # 3: Extract flow rate
                flow_rate = extract_flow_rate_from_file()
                if flow_rate is not None:
                    print(f"Measured Flow Rate: {flow_rate}")
                    # 4: Update Sheet
                    sheets_api.update_row(
                        batch_id=int(batch_id),
                        updates={"flow_rate": flow_rate},
                        sheet_name=SHEET_NAME
                    )
                    print(f"Updated sheet for batch {batch_id}")
                else:
                    print("Could not find flow rate in output file.")
            else:
                print(f"Simulation failed for batch {batch_id}")
                
        except Exception as e:
            print(f"Error processing batch {batch_id}: {e}")

    print("\nAll iterations complete.")

if __name__ == "__main__":
    main()
