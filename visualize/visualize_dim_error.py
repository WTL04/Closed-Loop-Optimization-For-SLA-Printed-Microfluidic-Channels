import sys
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contextual_opt.src.api.sheets_api import get_column

# IEEE-compliant styling: white background, no grid, paper context
sns.set_theme(style="ticks", context="paper")


def plot_dim_error(ax, df, label_prefix):
    x = df.index
    y_raw = df.iloc[:, 0].values
    y_best = np.minimum.accumulate(y_raw)

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

    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.set_ylabel("Cumulative Minimum NMSE", fontsize=12)
    ax.set_xlabel("Trial", fontsize=12)

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_ylim(bottom=0)

    # Subplot label
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


if __name__ == "__main__":
    sheet_names = ["Experiment Random Deltas", "Experiment Realistic Deltas"]

    dfs = []
    for sn in sheet_names:
        df = get_column("dim_error", sn)
        if df is None or df.empty:
            print(f"No data found for column 'dim_error' in '{sn}'")
            sys.exit()
        dfs.append(df)

    # IEEE-compliant figure settings
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8))
    fig.patch.set_facecolor("white")

    plot_dim_error(ax0, dfs[0], "a")
    ax0.set_title("Experiment Random Deltas", fontsize=13)

    plot_dim_error(ax1, dfs[1], "b")
    ax1.set_title("Experiment Realistic Deltas", fontsize=13)

    sns.despine(top=False, right=False, bottom=False, left=False)

    plt.tight_layout()
    plt.savefig("figures/Figure_2.pdf", bbox_inches="tight")
