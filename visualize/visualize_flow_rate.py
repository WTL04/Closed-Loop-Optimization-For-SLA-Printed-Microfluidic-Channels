import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contextual_opt.src.api.sheets_api import get_column

# 1. Enforce strict IEEE typography and Seaborn styling
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    }
)


def build_violin_data(df_raw, baseline_label, drift_label):
    baseline_data = df_raw.iloc[:80, 0].values
    drift_data = df_raw.iloc[80:, 0].values
    data = []
    for val in baseline_data:
        data.append({"Phase": baseline_label, "Flow Rate (mL/min)": val})
    for val in drift_data:
        data.append({"Phase": drift_label, "Flow Rate (mL/min)": val})
    return pd.DataFrame(data)


def plot_violin(ax, df_plot, label_prefix, title):
    ax.set_facecolor("white")

    slate_blue = "#4C72B0"
    dusty_orange = "#DD8452"

    sns.violinplot(
        data=df_plot,
        x="Flow Rate (mL/min)",
        y="Phase",
        palette=[slate_blue, dusty_orange],
        inner="quartile",
        linewidth=1.2,
        cut=0,
        ax=ax,
    )

    ax.set_ylabel("")
    ax.set_xlabel("Channel Flow Rate (mL/min)", fontsize=10)

    TARGET_FLOW = 0.1387
    ax.axvline(
        TARGET_FLOW,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        zorder=0,
    )

    ax.set_title(title, fontsize=11)

    ax.text(
        -0.2,
        1.05,
        f"({label_prefix})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )

    sns.despine(ax=ax)


if __name__ == "__main__":
    sheet_names = ["Experiment Random Deltas", "Experiment Realistic Deltas"]
    labels = [
        ("Initial Baseline\n(Trials 1-80)", "Random Noise\n(Trials 81-240)"),
        ("Initial Baseline\n(Trials 1-80)", "Realistic Drift\n(Trials 81-240)"),
    ]
    titles = ["Experiment Random Deltas", "Experiment Realistic Deltas"]

    dfs_plot = []
    for sn, (bl, dr) in zip(sheet_names, labels):
        df_raw = get_column("flow_rate", sn)
        if df_raw is None or df_raw.empty:
            print(f"No data found for column 'flow_rate' in '{sn}'")
            sys.exit()
        dfs_plot.append(build_violin_data(df_raw, bl, dr))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)
    fig.patch.set_facecolor("white")

    plot_violin(ax0, dfs_plot[0], "a", titles[0])
    plot_violin(ax1, dfs_plot[1], "b", titles[1])

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C72B0"),
        plt.Rectangle((0, 0), 1, 1, color="#DD8452"),
        plt.Line2D([0], [0], color="gray", linestyle=":", linewidth=1.0),
    ]
    labels = [
        "Initial Baseline\n(Trials 1-80)",
        "Dynamic Compensation\n(Trials 81-240)",
        "Nominal Target",
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.2),
    )

    plt.tight_layout()
    plt.savefig("figures/flow_rate_variance.pdf", format="pdf", bbox_inches="tight")
    plt.show()
