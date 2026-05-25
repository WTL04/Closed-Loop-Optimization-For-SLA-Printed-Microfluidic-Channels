from contextual_opt.src.pipeline.run import run_sequential
from contextual_opt.src.pipeline.config import SHEET_NAME

if __name__ == "__main__":
    for i in range(15):
        print("\n" + "=" * 60)
        print(f" {i} Channel Runs")
        print("=" * 60)

        # DEBUG: increase once confident context variables wont go in a weird unpredictable pattern
        num_channels = i

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
            sheet_name=SHEET_NAME,
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
