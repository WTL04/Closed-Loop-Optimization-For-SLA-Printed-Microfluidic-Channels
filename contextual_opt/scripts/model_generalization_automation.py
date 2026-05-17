"""
Runs sequential batches to help generalize the model.
All results will be appended to the Google Sheet.
Uses a checkpoint file to skip already-completed batches on restart.

Usage:
    python model_generalization_automation.py
    python model_generalization_automation.py --sheet-name "Experiment Realistic Deltas"
    python model_generalization_automation.py --sheet-name "Experiment Random Deltas" --case-dir cfd/channelCase_random
"""

import argparse
import json
import os

from contextual_opt.src.pipeline.run import run_sequential
from contextual_opt.src.pipeline.config import SHEET_NAME


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet-name", default=SHEET_NAME)
    parser.add_argument("--case-dir", default="cfd/channelCase")
    args = parser.parse_args()

    sheet_name = args.sheet_name
    case_dir = args.case_dir

    # Use sheet-specific checkpoint file to avoid conflicts between parallel instances
    safe_name = sheet_name.replace(" ", "_").replace('"', "")
    checkpoint_file = f"contextual_opt/src/data/batch_checkpoint_{safe_name}.json"

    def load_checkpoint():
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file) as f:
                return set(json.load(f))
        return set()

    def save_checkpoint(completed: set):
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, "w") as f:
            json.dump(list(completed), f)

    completed = load_checkpoint()

    print(f"Sheet name: {sheet_name}")
    print(f"Case dir: {case_dir}")
    print(f"Checkpoint file: {checkpoint_file}")

    for j in [10, 30, 50, 70]:
        for i in range(4):
            batch_key = f"{j}_{i}"

            if batch_key in completed:
                print(f"\nSkipping already-completed batch {batch_key}")
                continue

            print("\n" + "=" * 60)
            print(f" {j} Channel Runs, Batch {i + 1}/4")
            print("=" * 60)

            num_channels = j

            # Alternate temperature direction and layer thickness
            temp = "hot" if i in [0, 2] else "cold"
            layer_thickness_um = 100 if i in [0, 1] else 50

            testing = False
            append_to_sheets = True
            start_ambient = 80.0
            start_resin_age = 1.0
            resin_temp = 70.0

            results = run_sequential(
                sheet_name=sheet_name,
                num_channels=num_channels,
                temp=temp,
                layer_thickness_um=layer_thickness_um,
                testing=testing,
                append_to_sheets=append_to_sheets,
                start_ambient=start_ambient,
                start_resin_age=start_resin_age,
                resin_temp=resin_temp,
                case_dir=case_dir,
            )

            completed.add(batch_key)
            save_checkpoint(completed)
            print(f"\nBatch {batch_key} complete: {len(results)} channels run")
