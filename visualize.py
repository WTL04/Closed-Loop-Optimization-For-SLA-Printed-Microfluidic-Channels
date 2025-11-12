def visualize_control_chart(df_batches):
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(df_batches)+1), df_batches["cv"], marker="o", linestyle="-", label="Batch CVs")
    plt.axhline(df_batches["cv"].mean(), color="green", linestyle="--", label=f"Mean CV = {df_batches['cv'].mean():.3f}")
    plt.axhline(df_batches["cv"].mean() + 3*df_batches["cv"].std(), color="red", linestyle="--", label="UCL (Mean + 3σ)")
    plt.axhline(df_batches["cv"].mean() - 3*df_batches["cv"].std(), color="red", linestyle="--", label="LCL (Mean - 3σ)")
    plt.title("Control Chart for Batch CVs")
    plt.xlabel("Batch index")
    plt.ylabel("CV (std/mean)")
    plt.legend()
    plt.show()

def visualize_cv_drift_relationship(df_batches):
    for c in ["ambient_temp", "resin_temp", "resin_age"]:
        sns.regplot(
            x=df_batches[c],
            y=df_batches["cv"],
            scatter_kws={"s": 50, "alpha": 0.7},
            line_kws={"color": "red"},
        )
        plt.title(f"Flow Rate (CV) vs Drift Context ({c}) ")
        plt.xlabel(c)
        plt.ylabel("CV")
        plt.show()

def visualize_cv_histories(all_histories):
    plt.figure(figsize=(8, 4.5))
    for idx, hist in enumerate(all_histories):
        plt.plot(
            range(len(hist)), hist, marker="o", linestyle="-", label=f"Run {idx + 1}"
        )

        # highlight best point
        best_i = np.argmin(hist)
        plt.scatter(best_i, min(hist), color="red", marker="*", s=120)

    plt.title(
        "Flow Rate CV Improvement Across Iterations (Fixed Context: Warm Room Temp, 5-Day Resin)"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Coefficient of Variation (CV)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Independent Runs")
    plt.tight_layout()
    plt.show()

