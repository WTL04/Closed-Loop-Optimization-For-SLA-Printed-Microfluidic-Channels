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
    df = get_column("dim_error", "Experiment Random Deltas")

    if df is None or df.empty:
        print("No data found for column 'dim_error'")
        sys.exit()

    # Extract x and y arrays
    x = df.index
    y_raw = df.iloc[:, 0].values

    # Calculate the cumulative minimum (Best-Observed Value)
    y_best = np.minimum.accumulate(y_raw)

    # IEEE-compliant figure settings
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Muted palette hex codes
    slate_blue = "#4C72B0"

    # 1. Background line: Raw observations (faintly plotted to show exploration)
    ax.plot(
        x,
        y_raw,
        color="gray",
        linestyle="-",
        linewidth=0.8,
        alpha=0.3,
        label="Observed Evaluations",
    )

    # 2. Foreground line: Cumulative Minimum (Step plot)
    ax.plot(
        x,
        y_best,
        color=slate_blue,
        linestyle="-",
        linewidth=2.0,
        drawstyle="steps-post",  # Creates the classic BO staircase look
        label="Best-Observed NMSE",
    )

    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Academic y-axis label updated for the cumulative metric
    ax.set_ylabel("Cumulative Minimum NMSE", fontsize=12)
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
