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

def visualize_CV_Drift_relationship(df_batches):
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


