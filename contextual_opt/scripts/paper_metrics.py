import numpy as np
import pandas as pd
from contextual_opt.src.api.sheets_api import pullData

TARGET_FLOW = 0.13873  # mL/min (BASELINE_FLOW_RATE_ML_MIN from config)

random_df = pd.DataFrame(pullData(sheet_name="Experiment Random Deltas"))
realistic_df = pd.DataFrame(pullData(sheet_name="Experiment Realistic Deltas"))


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


random_df["dim_error"] = to_numeric(random_df["dim_error"])
random_df["flow_rate"] = to_numeric(random_df["flow_rate"])
realistic_df["dim_error"] = to_numeric(realistic_df["dim_error"])
realistic_df["flow_rate"] = to_numeric(realistic_df["flow_rate"])

baseline_rows = 80
random_active = random_df.iloc[baseline_rows:].copy()
realistic_active = realistic_df.iloc[baseline_rows:].copy()


def trials_to_converge(cummin_series, tol=0.01):
    final_val = cummin_series.iloc[-1]
    if final_val == 0:
        return len(cummin_series)
    threshold = final_val * (1 + tol)
    converged_idx = (cummin_series <= threshold).idxmax()
    return int(converged_idx - cummin_series.index[0]) + 1


# ============================================================
# SECTION A: Algorithmic Convergence and Adaptation
# ============================================================

print("=" * 70)
print("SECTION A: ALGORITHMIC CONVERGENCE AND ADAPTATION")
print("=" * 70)

for label, df, active in [
    ("Random Baseline", random_df, random_active),
    ("Realistic Profile", realistic_df, realistic_active),
]:
    print(f"\n--- {label} ---")

    dim = active["dim_error"].dropna().values
    cummin = np.minimum.accumulate(dim)

    # Best-observed NMSE
    best_nmse = cummin[-1]
    disp = f"{best_nmse:.4f}" if best_nmse >= 0.0001 else f"{best_nmse:.6f}"
    print(f"  Best-observed NMSE [X.XXXX]:       {disp}  (raw: {best_nmse:.2e})")

    # NMSE percent reduction
    initial_cummin = np.min(df["dim_error"].dropna().values[: baseline_rows + 1])
    final_cummin = np.min(df["dim_error"].dropna().values)
    reduction = (initial_cummin - final_cummin) / initial_cummin * 100
    print(f"  NMSE reduction [XX]%:               {reduction:.0f}%")

    # 10-trial rolling mean of dim_error (active phase only)
    active_dim = active["dim_error"].dropna()
    roll = active_dim.rolling(window=10, min_periods=1).mean()

    def fmt4(v):
        return f"{v:.4f}" if v >= 0.0001 else f"{v:.6f}"

    print(
        f"  Rolling mean initial [X.XXXX]:      {fmt4(roll.iloc[0])}  (raw: {roll.iloc[0]:.2e})"
    )
    print(
        f"  Rolling mean final [X.XXXX]:        {fmt4(roll.iloc[-1])}  (raw: {roll.iloc[-1]:.2e})"
    )

    # Median NMSE
    median_nmse = np.median(dim)
    disp = f"{median_nmse:.4f}" if median_nmse >= 0.0001 else f"{median_nmse:.6f}"
    print(f"  Median NMSE [X.XXXX]:               {disp}  (raw: {median_nmse:.2e})")

    # Trials to stable minimum
    cummin_series = pd.Series(cummin, index=active["dim_error"].dropna().index)
    ttc = trials_to_converge(cummin_series)
    print(f"  Trials to converge [XX]:            {ttc}")

# Realistic vs Random comparisons
rand_best = np.minimum.accumulate(random_active["dim_error"].dropna().values)[-1]
real_best = np.minimum.accumulate(realistic_active["dim_error"].dropna().values)[-1]
rand_ttc = trials_to_converge(
    pd.Series(np.minimum.accumulate(random_active["dim_error"].dropna().values))
)
real_ttc = trials_to_converge(
    pd.Series(np.minimum.accumulate(realistic_active["dim_error"].dropna().values))
)

pct_increase = (real_ttc - rand_ttc) / rand_ttc * 100
pct_of_baseline = (real_best - rand_best) / rand_best * 100

print(f"\n--- Cross-Experiment Comparisons ---")
print(f"  Trials increase vs baseline [XX]%:   {pct_increase:.0f}%")
print(f"  Realistic NMSE vs baseline [X.X]%:   {pct_of_baseline:.1f}%")

# ============================================================
# SECTION B: Hydrodynamic Validation
# ============================================================

print("\n" + "=" * 70)
print("SECTION B: HYDRODYNAMIC VALIDATION")
print("=" * 70)

for label, active in [
    ("Random Baseline", random_active),
    ("Realistic Profile", realistic_active),
]:
    print(f"\n--- {label} ---")

    fr = active["flow_rate"].dropna().values
    mean_fr = np.mean(fr)
    std_fr = np.std(fr)
    median_fr = np.median(fr)
    cv = std_fr / mean_fr * 100
    deviations = np.abs(fr - TARGET_FLOW) / TARGET_FLOW * 100

    print(f"  Mean flow rate [0.1XXX]:            {mean_fr:.4f}")
    print(f"  Std dev flow rate [0.0XXX]:         {std_fr:.4f}")
    print(f"  Median flow rate [0.1XXX]:          {median_fr:.4f}")
    print(f"  Coefficient of variation [X.X]%:    {cv:.1f}%")
    print(f"  Max deviation from target [X.X]%:   {deviations.max():.1f}%")

# Variance multiplier
rand_var = np.var(random_active["flow_rate"].dropna().values)
real_var = np.var(realistic_active["flow_rate"].dropna().values)
var_mult = real_var / rand_var
print(f"\n--- Variance Comparison ---")
print(f"  Variance factor [X.X]:               {var_mult:.1f}")

print("\n" + "=" * 70)
print("Done. Copy the printed values into the LaTeX placeholders.")
print("=" * 70)
