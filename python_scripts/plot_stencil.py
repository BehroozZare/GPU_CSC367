import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 16})

df = pd.read_csv("output/stencil.csv")

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

global_df = df[df["kernel_type"] == "global"].sort_values("block_size")
shared_df = df[df["kernel_type"] == "shared"].sort_values("block_size")

block_sizes = global_df["block_size"].values
x = np.arange(len(block_sizes))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars_g = ax.bar(x - width / 2, global_df["avg_time_us"].values, width, label="Global Memory")
bars_s = ax.bar(x + width / 2, shared_df["avg_time_us"].values, width, label="Shared Memory")

ax.bar_label(bars_g, fmt="%.0f", padding=3, fontsize=9)
ax.bar_label(bars_s, fmt="%.0f", padding=3, fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(block_sizes)
ax.set_xlabel("Threads per Block")
ax.set_ylabel("Time (μs)")
ax.set_title("1D Stencil: Global vs Shared Memory (R = 1)")
ax.legend()
ax.grid(axis="y")
clean_axes(ax)
fig.tight_layout()
fig.savefig("output/stencil_runtime.png", dpi=150)

plt.show()
