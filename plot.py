import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})

df = pd.read_csv("output/warp_divergence.csv")

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Plot 1: Block size 1024 only
fig1, ax1 = plt.subplots(figsize=(7, 5))
subset_1024 = df[df["block_size"] == 1024]
ax1.plot(subset_1024["tuning_parameter"], subset_1024["avg_time_us"], marker="o")
ax1.set_xticks(subset_1024["tuning_parameter"].values)
ax1.set_xlabel("Tuning Parameter")
ax1.set_ylabel("Time (μs)")
ax1.set_title("Block Size = 1024")
ax1.axvline(x=32, color="red", linestyle="--", alpha=0.5, label="Warp size (32)")
ax1.legend()
ax1.grid(True)
clean_axes(ax1)
fig1.tight_layout()
fig1.savefig("output/warp_divergence_1024.png", dpi=150)

# Plot 2: Block size 256 and 1024 overlaid
fig2, ax2 = plt.subplots(figsize=(7, 5))
for bs in [256, 1024]:
    subset = df[df["block_size"] == bs]
    ax2.plot(subset["tuning_parameter"], subset["avg_time_us"], marker="o", label=f"Block Size = {bs}")
ax2.set_xticks(df["tuning_parameter"].unique())
ax2.set_xlabel("Tuning Parameter")
ax2.set_ylabel("Time (μs)")
ax2.set_title("Block Size 256 vs 1024")
ax2.axvline(x=32, color="red", linestyle="--", alpha=0.5, label="Warp size (32)")
ax2.legend()
ax2.grid(True)
clean_axes(ax2)
fig2.tight_layout()
fig2.savefig("output/warp_divergence_256_vs_1024.png", dpi=150)

# Plot 3: CPU only
cpu_df = pd.read_csv("output/cpu_wrap.csv")
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.plot(cpu_df["tuning_parameter"], cpu_df["avg_time_us"], marker="o")
ax3.set_xticks(cpu_df["tuning_parameter"].values)
ax3.set_xlabel("Tuning Parameter")
ax3.set_ylabel("Time (μs)")
ax3.set_title("CPU Branch Divergence")
ax3.grid(True)
clean_axes(ax3)
fig3.tight_layout()
fig3.savefig("output/cpu_wrap_plot.png", dpi=150)

# Plot 4: CPU vs GPU (block size 1024)
fig4, ax4 = plt.subplots(figsize=(7, 5))
ax4.plot(cpu_df["tuning_parameter"], cpu_df["avg_time_us"], marker="o", label="CPU")
ax4.plot(subset_1024["tuning_parameter"], subset_1024["avg_time_us"], marker="o", label="GPU (Block Size = 1024)")
ax4.set_xticks(cpu_df["tuning_parameter"].values)
ax4.set_xlabel("Tuning Parameter")
ax4.set_ylabel("Time (μs)")
ax4.set_title("CPU vs GPU (Block Size 1024)")
ax4.legend()
ax4.grid(True)
clean_axes(ax4)
fig4.tight_layout()
fig4.savefig("output/cpu_vs_gpu_1024.png", dpi=150)

plt.show()
