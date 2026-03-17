# GPU Examples – CSC367

CUDA and CPU demo code for CSC367 at UofT. Includes warp-divergence, stride-access, and CPU baseline examples with profiling scripts for NVIDIA Nsight Compute.

## Repository Layout

```
demo/                  CUDA and C++ source files
  warp_example.cu
  stride_example.cu
  CPU_example.cpp
  utils/csv_writer.h
wolf_scripts/          SLURM job scripts for the Wolf cluster
  build.sh
  profile_warp_example.sh
plot.py                Plotting utility
run_*.sh               Local run helpers
profile_*.sh           Local profiling helpers
```

## Running on Wolf

### 1. Clone

```bash
ssh wolf
git clone https://github.com/BehroozZare/GPU_CSC367.git
cd GPU_CSC367
```

### 2. Build

Submit the build job **from the project root** (the scripts use `SLURM_SUBMIT_DIR` to locate the source):

```bash
sbatch wolf_scripts/build.sh
```

Executables (`warp_example`, `stride_example`, `cpu_example`) are placed in `~/GPU_CSC367/build/`. Check the build log:

```bash
cat build_<jobid>.out
```

If you need a clean rebuild:

```bash
rm -rf ~/GPU_CSC367/build
sbatch wolf_scripts/build.sh
```

### 3. Run

Run an executable directly on a compute node via `srun`:

```bash
srun -p csc367-compute --gres=gpu ~/GPU_CSC367/build/warp_example <tuning_parameter> <block_size>
```

### 4. Profile

Profile the warp example with Nsight Compute:

```bash
sbatch wolf_scripts/profile_warp_example.sh <tuning_parameter> <block_size>
```

For example:

```bash
sbatch wolf_scripts/profile_warp_example.sh 16 256
```

Reports are saved to `~/GPU_CSC367/prof_reports/` as `.ncu-rep` files. SLURM output goes to `profiling_<jobid>.out` in the directory you submitted from.

### 5. View Reports

Copy `.ncu-rep` files to your local machine and open them in NVIDIA Nsight Compute:

```bash
scp wolf:~/GPU_CSC367/prof_reports/*.ncu-rep .
```

## Build Targets

| Target | Source | Description |
|---|---|---|
| `warp_example` | `demo/warp_example.cu` | Warp-divergence experiment |
| `stride_example` | `demo/stride_example.cu` | Strided memory access experiment |
| `cpu_example` | `demo/CPU_example.cpp` | CPU baseline (OpenMP) |
