#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "csv_writer.h"


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Naive 1D stencil (R=1): each thread reads 3 elements from global memory.
// Neighboring threads redundantly load overlapping elements.
__global__ void stencil_global(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    float left   = (gid > 0)     ? input[gid - 1] : 0.0f;
    float center =                  input[gid];
    float right  = (gid < N - 1) ? input[gid + 1] : 0.0f;

    output[gid] = (left + center + right) / 3.0f;
}

// Shared-memory 1D stencil (R=1): threads cooperatively load a tile
// plus a 1-element halo on each side, then compute from shared memory.
//
// Shared memory layout (blockDim.x + 2 floats):
//   [left_halo | ---- tile (blockDim.x) ---- | right_halo]
//    smem[0]     smem[1] ... smem[blockDim.x]  smem[blockDim.x+1]
__global__ void stencil_shared(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N) {
    extern __shared__ float smem[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    // Load center element into tile region
    smem[lid + 1] = (gid < N) ? input[gid] : 0.0f;

    // First thread in block loads left halo
    if (lid == 0) {
        smem[0] = (gid > 0) ? input[gid - 1] : 0.0f;
    }

    // Last thread in block loads right halo
    if (lid == blockDim.x - 1) {
        smem[blockDim.x + 1] = (gid + 1 < N) ? input[gid + 1] : 0.0f;
    }

    __syncthreads();

    if (gid >= N) return;

    output[gid] = (smem[lid] + smem[lid + 1] + smem[lid + 2]) / 3.0f;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <threads_per_block>\n", argv[0]);
        return 1;
    }

    int threads_per_block = std::atoi(argv[1]);

    const int N = 10000000;
    const int NUM_ITER = 10;

    std::vector<float> h_input(N);
    std::vector<float> h_out_global(N);
    std::vector<float> h_out_shared(N);
    float *d_input  = nullptr;
    float *d_output = nullptr;

    for (int i = 0; i < N; i++)
        h_input[i] = static_cast<float>(i);

    printf("Array size     : %lu MB (%d elements)\n",
           (unsigned long)(N * sizeof(float)) / 1024 / 1024, N);
    printf("Block size     : %d\n", threads_per_block);

    GPU_CHECK(cudaMalloc(&d_input,  N * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    GPU_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(threads_per_block);
    dim3 grid((N + threads_per_block - 1) / threads_per_block);

    // ---- Benchmark: global memory kernel ----
    stencil_global<<<grid, block>>>(d_input, d_output, N);
    GPU_CHECK(cudaDeviceSynchronize());

    auto start_g = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITER; i++) {
        stencil_global<<<grid, block>>>(d_input, d_output, N);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
    }
    auto end_g = std::chrono::high_resolution_clock::now();
    long long avg_global = std::chrono::duration_cast<std::chrono::microseconds>(end_g - start_g).count() / NUM_ITER;

    GPU_CHECK(cudaMemcpy(h_out_global.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Benchmark: shared memory kernel ----
    size_t smem_bytes = (threads_per_block + 2) * sizeof(float);

    stencil_shared<<<grid, block, smem_bytes>>>(d_input, d_output, N);
    GPU_CHECK(cudaDeviceSynchronize());

    auto start_s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITER; i++) {
        stencil_shared<<<grid, block, smem_bytes>>>(d_input, d_output, N);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
    }
    auto end_s = std::chrono::high_resolution_clock::now();
    long long avg_shared = std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count() / NUM_ITER;

    GPU_CHECK(cudaMemcpy(h_out_shared.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Correctness check ----
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = std::fabs(h_out_global[i] - h_out_shared[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff       : %e\n", max_diff);
    printf("Global memory  : %lld us\n", avg_global);
    printf("Shared memory  : %lld us\n", avg_shared);

    // ---- Write CSV ----
    CsvWriter writer("/home/behrooz/Desktop/CSC367/GPU/output/stencil.csv",
                     {"timestamp",
                      "kernel_type",
                      "num_elements",
                      "block_size",
                      "grid_size",
                      "avg_time_us"});

    if (writer.good()) {
        auto now = std::chrono::system_clock::now();
        auto now_t = std::chrono::system_clock::to_time_t(now);
        std::string ts = std::to_string(static_cast<long long>(now_t));

        writer.write_row({ts, "global", std::to_string(N),
                          std::to_string(threads_per_block),
                          std::to_string(grid.x),
                          std::to_string(avg_global)});

        writer.write_row({ts, "shared", std::to_string(N),
                          std::to_string(threads_per_block),
                          std::to_string(grid.x),
                          std::to_string(avg_shared)});
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
