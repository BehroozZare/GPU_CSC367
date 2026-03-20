#!../../Last_Project/homa_pyenv/bin/python
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})

df = pd.read_csv("output/warp_stride.csv")

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df["stride"], df["avg_time_us"], marker="o")
for _, row in df.iterrows():
    ax.annotate(f'{row["avg_time_us"]:.0f}', (row["stride"], row["avg_time_us"]),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
ax.set_xscale("log", base=2)
ax.set_xticks(df["stride"].values)
ax.set_xticklabels(df["stride"].values, rotation=45, ha="right")
import numpy as np
y_max = df["avg_time_us"].max()
ax.set_yticks(np.arange(0, y_max + 20, 20))
ax.set_xlabel("Stride")
ax.set_ylabel("Time (μs)")
ax.set_title("Runtime vs Stride")
ax.grid(True)
clean_axes(ax)
fig.tight_layout()
fig.savefig("output/stride_runtime.png", dpi=150)

plt.show()
