from sklearn.linear_model import LinearRegression
from contextual_opt.src.api.sheets_api import pullData
import pandas as pd

df = pullData("Experiment Realistic Deltas", verbose=False)
df = df[df["flow_rate"] > 0]

X = df[["ambient_temp", "resin_age"]].values
for dim in ["delta_length_um", "delta_width_um", "delta_height_um"]:
    y = df[dim].values
    reg = LinearRegression().fit(X, y)
    print(
        f"{dim}: temp={reg.coef_[0]:.4f}, age={reg.coef_[1]:.4f}, R²={reg.score(X, y):.3f}"
    )
