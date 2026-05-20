"""
Runs sequential batches to help generalize the model.
All results will be appended to the Google Sheet.
On restart, recovers position from sheet data.

Usage:
    python model_generalization_automation.py
    python model_generalization_automation.py --sheet-name "Experiment Realistic Deltas"
    python model_generalization_automation.py --sheet-name "Experiment Random Deltas" --case-dir cfd/channelCase_random
"""

import argparse

from contextual_opt.src.pipeline.run import run_sequential
from contextual_opt.src.pipeline.config import SHEET_NAME
from contextual_opt.src.api.sheets_api import pullData

# Number of initial data rows in the sheet before automation began
BASE_ROWS = 80

# Block definitions: (j, capacity_per_batch, total_capacity)
# Must match the outer loop in __main__
BLOCKS = [(10, 10, 40), (30, 30, 120)]


def recover_state(sheet_name):
    """
    Query the sheet to determine current batch position and context state.

    Returns:
        (start_j, start_i, channels_remaining, last_ambient, last_resin_age)
        - start_j: block size to resume (10, 30, or 50)
        - start_i: batch index to resume (0-3)
        - channels_remaining: channels left in this batch (0 = batch complete)
        - last_ambient: ambient_temp from last sheet row (for mid-batch resume)
        - last_resin_age: resin_age from last sheet row (for mid-batch resume)
    """
    df = pullData(sheet_name=sheet_name, verbose=False)
    total_rows = len(df)
    automation_trials = total_rows - BASE_ROWS

    if automation_trials <= 0:
        return 10, 0, 0, 80.0, 1.0

    # Determine which block we're in
    accumulated = 0
    for j, batch_size, block_capacity in BLOCKS:
        if automation_trials < accumulated + block_capacity:
            # We're in this block
            trials_in_block = automation_trials - accumulated
            start_i = trials_in_block // batch_size
            channels_in_batch = trials_in_block % batch_size
            channels_remaining = batch_size - channels_in_batch

            # Read last row for context drift
            if not df.empty and channels_in_batch > 0:
                last = df.iloc[-1]
                last_ambient = float(last["ambient_temp"])
                last_resin_age = float(last["resin_age"])
            else:
                last_ambient = 80.0
                last_resin_age = 1.0

            # If batch is exactly full (channels_in_batch == 0), next batch starts fresh
            if channels_in_batch == 0:
                last_ambient = 80.0
                last_resin_age = 1.0

            return (
                j,
                start_i,
                channels_remaining if channels_in_batch > 0 else 0,
                last_ambient,
                last_resin_age,
            )

        accumulated += block_capacity

    # All blocks complete — nothing to do
    return BLOCKS[-1][0], 3, 0, 80.0, 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet-name", default=SHEET_NAME)
    parser.add_argument("--case-dir", default="cfd/channelCase")
    args = parser.parse_args()

    sheet_name = args.sheet_name
    case_dir = args.case_dir

    print(f"Sheet name: {sheet_name}")
    print(f"Case dir: {case_dir}")

    # Recover state from sheet
    start_j, start_i, channels_remaining, recovered_ambient, recovered_resin_age = (
        recover_state(sheet_name)
    )

    print(
        f"Recovered state: block={start_j}, batch={start_i}, remaining_in_batch={channels_remaining}"
    )
    print(
        f"Last context: ambient={recovered_ambient}°F, resin_age={recovered_resin_age}hr"
    )

    # Determine which block index to start from
    block_indices = [b[0] for b in BLOCKS]
    start_block_idx = block_indices.index(start_j)

    for b_idx in range(start_block_idx, len(BLOCKS)):
        j = BLOCKS[b_idx][0]
        first_i = start_i if j == start_j else 0

        for i in range(first_i, 4):
            print("\n" + "=" * 60)
            print(f" {j} Channel Runs, Batch {i + 1}/4")
            print("=" * 60)

            # If resuming a partial batch, run only remaining channels
            if j == start_j and i == start_i and channels_remaining > 0:
                num_channels = channels_remaining
                start_ambient = recovered_ambient
                start_resin_age = recovered_resin_age
                print(f"Resuming mid-batch: {num_channels} channels remaining")
            else:
                num_channels = j
                start_ambient = 80.0
                start_resin_age = 1.0

            # Alternate temperature direction and layer thickness
            temp = "hot" if i in [0, 2] else "cold"
            layer_thickness_um = 100 if i in [0, 1] else 50

            testing = False
            append_to_sheets = True
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

            print(
                f"\nBatch {j} channels, {i + 1}/4 complete: {len(results)} channels run"
            )
