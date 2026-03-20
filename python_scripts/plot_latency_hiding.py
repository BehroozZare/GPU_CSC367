import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})

df = pd.read_csv("../output/latency_hiding.csv")

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df["block_size"], df["avg_time_us"], marker="o")
ax.set_xticks(df["block_size"].values)
ax.set_xlabel("Threads per Block")
ax.set_ylabel("Time (μs)")
ax.set_title("Latency Hiding: Pointer-Chasing Kernel")
ax.axvline(x=32, color="red", linestyle="--", alpha=0.5, label="1 warp")
ax.legend()
ax.grid(True)
clean_axes(ax)
fig.tight_layout()
fig.savefig("../output/latency_hiding.png", dpi=150)

plt.show()
