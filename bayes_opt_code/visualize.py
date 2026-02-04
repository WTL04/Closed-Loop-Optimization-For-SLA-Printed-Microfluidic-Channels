# visualize.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid", rc={"grid.linestyle": "--", "grid.alpha": 0.6})


def visualize_control_chart(df_batches, save_path=None):
    """
    Control chart for batch CVs.
    df_batches: pd.DataFrame with column "cv"
    save_path: optional path to save the figure (e.g. "control_chart.png")
    """
    if df_batches is None or "cv" not in df_batches or len(df_batches) == 0:
        print("[visualize_control_chart] No CV data available.")
        return

    cvs = np.array(df_batches["cv"].astype(float))
    x = np.arange(1, len(cvs) + 1)

    mean_cv = cvs.mean()
    std_cv = cvs.std(ddof=0)
    ucl = mean_cv + 3 * std_cv
    lcl = max(mean_cv - 3 * std_cv, 0.0)

    plt.figure(figsize=(10, 5))
    plt.plot(x, cvs, marker="o", linestyle="-", label="Batch CVs")
    plt.axhline(mean_cv, color="green", linestyle="--", label=f"Mean CV = {mean_cv:.4f}")
    plt.axhline(ucl, color="red", linestyle="--", label=f"UCL (Mean + 3σ) = {ucl:.4f}")
    plt.axhline(lcl, color="red", linestyle="--", label=f"LCL (Mean - 3σ) = {lcl:.4f}")
    plt.title("Control Chart for Batch CVs")
    plt.xlabel("Batch index")
    plt.ylabel("CV (std/mean)")
    plt.legend(loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def visualize_cv_drift_relationship(df_batches, save_path_prefix=None):
    """
    Create scatter + regression plots of CV vs each drift/context variable.
    df_batches: pd.DataFrame with columns ["ambient_temp", "resin_temp", "resin_age", "cv"]
    save_path_prefix: optional prefix; each plot saved as '{prefix}_{col}.png'
    """
    if df_batches is None or "cv" not in df_batches or len(df_batches) == 0:
        print("[visualize_cv_drift_relationship] No CV data available.")
        return

    for c in ["ambient_temp", "resin_temp", "resin_age"]:
        if c not in df_batches:
            print(f"[visualize_cv_drift_relationship] Column '{c}' not found, skipping.")
            continue

        plt.figure(figsize=(7, 4.2))
        sns.regplot(
            x=df_batches[c].astype(float),
            y=df_batches["cv"].astype(float),
            scatter_kws={"s": 50, "alpha": 0.7},
            line_kws={"color": "red"},
        )
        plt.title(f"Flow Rate (CV) vs Drift Context ({c})")
        plt.xlabel(c)
        plt.ylabel("CV")
        plt.tight_layout()

        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_{c}.png", dpi=200)
        plt.show()


def visualize_model_convergence(all_histories, save_path=None):
    """
    Plot raw CV per iteration for each run and mark the lowest point (star).
    all_histories: list of lists (each inner list = raw CV trajectory)
    """
    if not all_histories:
        print("[visualize_model_convergence] No history data provided.")
        return

    plt.figure(figsize=(8, 4.5))

    for idx, hist in enumerate(all_histories):
        if hist is None or len(hist) == 0:
            continue

        hist_arr = np.array(hist, dtype=float)
        iters = np.arange(len(hist_arr))

        plt.plot(iters, hist_arr, marker="o", linestyle="-", label=f"Run {idx + 1}")

        # highlight best (lowest) point
        best_i = int(np.argmin(hist_arr))
        best_val = float(hist_arr[best_i])
        plt.scatter(best_i, best_val, color="red", marker="*", s=140, edgecolor="black", zorder=6)
        # small annotation slightly above the star
        plt.annotate(f"{best_val:.4f}", (best_i, best_val), xytext=(6, 6), textcoords="offset points", fontsize=9)

    plt.title("Flow Rate CV Improvement Across Iterations (Fixed Context: Warm Room Temp, 5-Day Resin)")
    plt.xlabel("Iteration")
    plt.ylabel("Coefficient of Variation (CV)")
    plt.grid(True)
    plt.legend(title="Independent Runs")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def visualize_model_convergence(all_histories):
    """Raw CV only (clean)."""
    plt.figure(figsize=(9, 5))

    for run_idx, raw_list in enumerate(all_histories, start=1):
        raw = np.array(raw_list, dtype=float)
        iters = np.arange(len(raw))

        plt.plot(iters, raw, marker="o", label=f"Run {run_idx}")

        min_idx = int(np.argmin(raw))
        min_val = float(raw[min_idx])
        plt.scatter(min_idx, min_val, marker="*", s=240, edgecolor="black", zorder=6)
        plt.text(min_idx, min_val, f"  {min_val:.4f}", va="bottom")

    plt.xlabel("Iteration")
    plt.ylabel("CV")
    plt.title("CV per Iteration (Lowest Point Marked)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
