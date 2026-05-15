"""
Runs 120 trials in total to help generalize the model
All results will be appended to the google sheets

- 30 runs (ambient_temp: increase, layer_thickness: 100um)
- 30 runs (ambient_temp: decrease, layer_thickness: 100um)
- 30 runs (ambient_temp: increase, layer_thickness: 50um)
- 30 runs (ambient_temp: decrease, layer_thickness: 50um)

Each batch starts fresh with new CBO for independent exploration.
"""

from contextual_opt.src.pipeline.run import run_sequential
from contextual_opt.src.pipeline.config import SHEET_NAME

if __name__ == "__main__":
    sheet_name = SHEET_NAME

    # 10 runs, 30, runs, 50 runs, 70 runs
    # TEST: only 10 number of channels per batch
    for j in [10]:
        for i in range(4):
            print("\n" + "=" * 60)
            print(f" {j} Channel Runs, Batch {i + 1}/4")
            print("=" * 60)

            # DEBUG: increase once confident context variables wont go in a weird unpredictable pattern
            num_channels = j

            # set temperature direction, alternate
            temp = "hot" if i in [0, 2] else "cold"

            # set layer thickness
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
            )

            print(f"\nBatch {i + 1} complete: {len(results)} channels run")
