# GPU Examples – CSC367

CUDA and CPU demo code for CSC367 at UofT. Includes warp-divergence, stride-access, latency-hiding, stencil (shared memory), and CPU baseline examples with profiling scripts for NVIDIA Nsight Compute.

## Repository Layout

```
demo/                         CUDA and C++ source files
  warp_example.cu
  stride_example.cu
  simple_copy.cu
  latency_hiding_example.cu
  stencil_example.cu
  CPU_example.cpp
  utils/csv_writer.h
local_scripts/                Local run and profiling helpers
  run_warp_example.sh
  run_stride_example.sh
  run_cpu_example.sh
  run_latency_hiding_example.sh
  run_simple_copy.sh
  run_stencil_example.sh
  profile_warp_example.sh
  profile_stride_example.sh
  profile_latency_hiding_example.sh
  profile_stencil_example.sh
python_scripts/               Plotting scripts
  plot_warp.py
  plot_stride.py
  plot_latency_hiding.py
  plot_stencil.py
snippets/                     Lecture/reference snippets
  slides_snippets.txt
  profiling_snippets.txt
  latency_hiding_snippets.txt
  stencil_snippets.txt
wolf_scripts/                 SLURM job scripts for the Wolf cluster
  build.sh
  profile_warp_example.sh
output/                       Generated CSVs and plots
prof_reports/                 Nsight Compute reports (.ncu-rep)
build/                        CMake build output
```

## Running Locally

### Build

```bash
cmake -B build -DCUDA_ARCHITECTURES="86"
cmake --build build
```

### Run

Use the helper scripts in `local_scripts/`:

```bash
local_scripts/run_warp_example.sh
local_scripts/run_stride_example.sh
local_scripts/run_latency_hiding_example.sh
local_scripts/run_simple_copy.sh
local_scripts/run_stencil_example.sh
local_scripts/run_cpu_example.sh
```

### Profile

```bash
local_scripts/profile_warp_example.sh <tuning_parameter> <block_size>
local_scripts/profile_stride_example.sh <stride> <block_size>
local_scripts/profile_latency_hiding_example.sh <chain_length> <threads_per_block>
local_scripts/profile_stencil_example.sh <threads_per_block>
```

### Plot

Run plotting scripts from the project root (they read from `output/`):

```bash
python python_scripts/plot_warp.py
python python_scripts/plot_stride.py
python python_scripts/plot_latency_hiding.py
python python_scripts/plot_stencil.py
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

Executables are placed in `~/GPU_CSC367/build/`. Check the build log:

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
| `simple_copy` | `demo/simple_copy.cu` | Simple device memory copy |
| `latency_hiding_example` | `demo/latency_hiding_example.cu` | Latency hiding via occupancy |
| `stencil_example` | `demo/stencil_example.cu` | 1D stencil: global vs shared memory |
| `cpu_example` | `demo/CPU_example.cpp` | CPU baseline (OpenMP) |
