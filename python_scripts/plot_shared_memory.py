import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})

df = pd.read_csv("output/shared_memory_transpose.csv")

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

latest_ts = df["timestamp"].max()
latest = df[df["timestamp"] == latest_ts].copy()

label_map = {
    "naive": "No Shared Memory",
    "shared_mem": "Shared Memory",
    "shared_mem_padded": "Shared Memory\n+ No Bank Conflict",
}

latest["label"] = latest["kernel_type"].map(label_map)
matrix_size = latest["matrix_size"].iloc[0]

def make_bar_plot(data, order, filename, title_suffix=""):
    subset = data.set_index("kernel_type").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(subset["label"], subset["avg_time_us"], width=0.5)
    for bar, val in zip(bars, subset["avg_time_us"]):
        ax.annotate(f"{val:.0f} \u00b5s",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=13)
    ax.set_ylabel("Time (\u00b5s)")
    ax.set_title(f"Matrix Transpose ({matrix_size}\u00d7{matrix_size}){title_suffix}")
    ax.grid(axis="y", alpha=0.3)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    return fig

make_bar_plot(latest, ["naive", "shared_mem"],
              "output/shared_memory_transpose_basic.png",
              " -- Shared Memory")

make_bar_plot(latest, ["naive", "shared_mem", "shared_mem_padded"],
              "output/shared_memory_transpose.png",
              " -- Bank Conflict Fix")

plt.show()
