from contextual_opt.src.pipeline.delta_loaders import generate_realistic_deltas
import numpy as np

print("=== Delta range check ===")
for temp in [60, 75, 100]:
    for age in [0, 87, 174]:
        samples = [
            generate_realistic_deltas(ambient_temp=temp, resin_age_hours=age)
            for _ in range(2000)
        ]
        widths = [s["width"] for s in samples]
        print(
            f"temp={temp}°F age={age}h: "
            f"mean={np.mean(widths):.2f} "
            f"std={np.std(widths):.2f} "
            f"max={np.max(widths):.2f} "
            f"min={np.min(widths):.2f}"
        )
