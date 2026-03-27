import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 16})

df = pd.read_csv("../output/stream_concurrency.csv")

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

labels = [f"{it//1000}k" for it in df["iterations"]]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars_serial = ax.bar(x - width / 2, df["serial_time_ms"], width, label="Serial (default stream)")
bars_concurrent = ax.bar(x + width / 2, df["concurrent_time_ms"], width, label="Concurrent (2 streams)")

for i, sp in enumerate(df["speedup"]):
    y = max(df["serial_time_ms"].iloc[i], df["concurrent_time_ms"].iloc[i])
    ax.text(x[i], y + 1.5, f"{sp:.2f}x", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_xlabel("Iterations per element")
ax.set_ylabel("Time (ms)")
ax.set_title("CUDA Streams: Serial vs Concurrent Kernel Execution")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, axis="y")
clean_axes(ax)
fig.tight_layout()
fig.savefig("../output/stream_concurrency.png", dpi=150)

plt.show()
