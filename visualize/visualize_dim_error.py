import sys
from pathlib import Path

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contextual_opt.src.pipeline.config import NOMINAL_DIMENSIONS

sns.set_theme(style="ticks", context="paper")


def plot_dim_error_active(ax, df, label_prefix):
    x = df["trial_num"]
    y_best = df["cummin_nmse"]
    y_raw = df["dim_error"]
    y_rolling = df["rolling_mean"]

    ax.set_facecolor("white")
    slate_blue = "#4C72B0"

    ax.plot(
        x,
        y_raw,
        color="gray",
        linestyle="-",
        linewidth=0.8,
        alpha=0.3,
        label="Observed Evaluations",
    )

    ax.plot(
        x,
        y_best,
        color=slate_blue,
        linestyle="-",
        linewidth=2.0,
        drawstyle="steps-post",
        label="Best-Observed NMSE",
    )

    ax.plot(
        x,
        y_rolling,
        color="steelblue",
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
        label="Rolling Mean (10-trial)",
    )

    ax.set_ylabel("NMSE", fontsize=12)
    ax.set_xlabel("Active Trial", fontsize=12)

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_ylim(bottom=0)

    ax.legend(loc="upper right", frameon=True, fontsize=10)

    ax.text(
        -0.15,
        1.05,
        f"({label_prefix})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def load_experiment(sheet_name):
    try:
        from contextual_opt.src.api.sheets_api import pullData

        df = pullData(sheet_name=sheet_name, verbose=False)
        if df is not None and not df.empty:
            df["channel_length_um"] = pd.to_numeric(
                df["channel_length_um"], errors="coerce"
            )
            df["dim_error"] = pd.to_numeric(df["dim_error"], errors="coerce")
            return df
    except Exception as e:
        print(f"pullData failed for '{sheet_name}': {e}")

    csv_map = {
        "Experiment Random Deltas": "contextual_opt/datasets/experiment_random_deltas.csv",
        "Experiment Realistic Deltas": "contextual_opt/datasets/experiment_realistic_deltas.csv",
    }
    path = csv_map.get(sheet_name)
    if path and Path(path).exists():
        return pd.read_csv(path)

    return None


if __name__ == "__main__":
    sheet_names = ["Experiment Random Deltas", "Experiment Realistic Deltas"]
    nominal_length = NOMINAL_DIMENSIONS["length"]

    dfs = []
    for sn in sheet_names:
        df = load_experiment(sn)
        if df is None or df.empty:
            print(f"No data found for '{sn}'")
            sys.exit()

        # Filter to active Ax trials only (exclude historical warm-start rows
        # which have nominal channel_length_um == 40000)
        df_active = df[df["channel_length_um"] != nominal_length].copy()

        if df_active.empty:
            print(f"No active trials found in '{sn}'")
            sys.exit()

        df_active["trial_num"] = range(1, len(df_active) + 1)
        df_active["cummin_nmse"] = df_active["dim_error"].cummin()
        df_active["rolling_mean"] = (
            df_active["dim_error"].rolling(window=10, min_periods=1).mean()
        )
        dfs.append(df_active)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8))
    fig.patch.set_facecolor("white")

    plot_dim_error_active(ax0, dfs[0], "a")
    plot_dim_error_active(ax1, dfs[1], "b")

    sns.despine(top=False, right=False, bottom=False, left=False)

    plt.tight_layout()
    plt.savefig("figures/dim_error_convergence.pdf", bbox_inches="tight")
    plt.show()
