import pandas as pd
from contextual_opt.src.api.sheets_api import pullData

df = pd.DataFrame(pullData(sheet_name="Experiment Realistic Deltas"))

hot = df[df["ambient_temp"] > 85]
cold = df[df["ambient_temp"] < 70]

print(f"Hot rows: {len(hot)}, Cold rows: {len(cold)}")

print("\nHot block suggested geometry:")
print(hot[["channel_length_um", "channel_width_um", "channel_height_um"]].describe())

print("\nCold block suggested geometry:")
print(cold[["channel_length_um", "channel_width_um", "channel_height_um"]].describe())

print("\nMean difference (hot - cold):")
print(
    hot[["channel_length_um", "channel_width_um", "channel_height_um"]].mean()
    - cold[["channel_length_um", "channel_width_um", "channel_height_um"]].mean()
)
