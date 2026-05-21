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

if __name__ == "__main__":
    df = get_column("flow_rate", "Experiment Realistic Deltas")

    if df is None or df.empty:
        print("No data found for column 'dim_error'")
        sys.exit()

    # IEEE-compliant figure settings
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Muted palette hex codes
    slate_blue = "#4C72B0"
    dusty_orange = "#DD8452"

    # Data line: solid with markers for grayscale distinction
    ax.plot(
        df.index,
        df.iloc[:, 0].values,
        color=slate_blue,
        linestyle="-",
        linewidth=1.2,
        marker="o",
        markersize=3,
        markerfacecolor=slate_blue,
        markeredgecolor=slate_blue,
        alpha=0.8,  # Adds that slight transparency you liked
        label="NMSE",
    )

    # Trend line: dashed without markers for grayscale distinction
    x = df.index
    y = df.iloc[:, 0].values
    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x + intercept
    ax.plot(
        x,
        trend,
        color=dusty_orange,
        linestyle="--",
        linewidth=1.5,
        label="Linear Trend",
    )

    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Academic y-axis label
    ax.set_ylabel("Normalized Mean Squared Error (NMSE)", fontsize=12)
    ax.set_xlabel("Trial", fontsize=12)

    # Scientific notation for y-axis
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Force y-axis to start at 0
    ax.set_ylim(bottom=0)

    # IEEE-compliant spines: all four sides visible
    sns.despine(top=False, right=False, bottom=False, left=False)

    plt.tight_layout()
    plt.show()
