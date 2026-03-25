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

#define TILE_DIM 32

#define INDX(row, col, ld) ((row) * (ld) + (col))

// Naive matrix transpose using only global memory.
// Reads are coalesced (consecutive threadIdx.x -> consecutive columns),
// but writes are uncoalesced (consecutive threadIdx.x -> stride-ld apart).
__global__ void naive_transpose(const float* __restrict__ a,
                                float* __restrict__ c,
                                int m) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < m && col < m)
        c[INDX(col, row, m)] = a[INDX(row, col, m)];
}

// Shared-memory matrix transpose.
// A 32x32 tile is loaded into shared memory so that both the global
// read and the global write are coalesced.
// Note: the store into shared memory has a 32-way bank conflict because
// consecutive threads (threadIdx.x) index the first dimension of a
// row-major 2D array, giving stride-32 access across banks.
__global__ void shared_mem_transpose(const float* __restrict__ a,
                                     float* __restrict__ c,
                                     int m) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int tileCol = blockDim.x * blockIdx.x;
    int tileRow = blockDim.y * blockIdx.y;

    int col = tileCol + threadIdx.x;
    int row = tileRow + threadIdx.y;

    if (row < m && col < m)
        tile[threadIdx.x][threadIdx.y] = a[INDX(row, col, m)];

    __syncthreads();

    int outCol = tileRow + threadIdx.x;
    int outRow = tileCol + threadIdx.y;

    if (outRow < m && outCol < m)
        c[INDX(outRow, outCol, m)] = tile[threadIdx.y][threadIdx.x];
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int M = std::atoi(argv[1]);
    const int NUM_ITER = 10;
    long long total_elements = (long long)M * M;

    std::vector<float> h_a(total_elements);
    std::vector<float> h_out_naive(total_elements);
    std::vector<float> h_out_shared(total_elements);
    float *d_a = nullptr;
    float *d_c = nullptr;

    for (long long i = 0; i < total_elements; i++)
        h_a[i] = static_cast<float>(i);

    printf("Matrix size    : %d x %d (%.1f MB)\n",
           M, M, (double)(total_elements * sizeof(float)) / 1024.0 / 1024.0);

    GPU_CHECK(cudaMalloc(&d_a, total_elements * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_c, total_elements * sizeof(float)));
    GPU_CHECK(cudaMemcpy(d_a, h_a.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // ---- Benchmark: naive transpose (global memory only) ----
    naive_transpose<<<grid, block>>>(d_a, d_c, M);
    GPU_CHECK(cudaDeviceSynchronize());

    auto start_n = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITER; i++) {
        naive_transpose<<<grid, block>>>(d_a, d_c, M);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
    }
    auto end_n = std::chrono::high_resolution_clock::now();
    long long avg_naive = std::chrono::duration_cast<std::chrono::microseconds>(end_n - start_n).count() / NUM_ITER;

    GPU_CHECK(cudaMemcpy(h_out_naive.data(), d_c, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Benchmark: shared memory transpose (with bank conflicts) ----
    shared_mem_transpose<<<grid, block>>>(d_a, d_c, M);
    GPU_CHECK(cudaDeviceSynchronize());

    auto start_s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITER; i++) {
        shared_mem_transpose<<<grid, block>>>(d_a, d_c, M);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
    }
    auto end_s = std::chrono::high_resolution_clock::now();
    long long avg_shared = std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count() / NUM_ITER;

    GPU_CHECK(cudaMemcpy(h_out_shared.data(), d_c, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Correctness check ----
    float max_diff = 0.0f;
    for (long long i = 0; i < total_elements; i++) {
        float diff = std::fabs(h_out_naive[i] - h_out_shared[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff       : %e\n", max_diff);
    printf("Naive (global) : %lld us\n", avg_naive);
    printf("Shared memory  : %lld us\n", avg_shared);

    // ---- Write CSV ----
    CsvWriter writer("/home/behrooz/Desktop/CSC367/GPU/output/shared_memory_transpose.csv",
                     {"timestamp",
                      "kernel_type",
                      "matrix_size",
                      "block_dim",
                      "grid_dim",
                      "avg_time_us"});

    if (writer.good()) {
        auto now = std::chrono::system_clock::now();
        auto now_t = std::chrono::system_clock::to_time_t(now);
        std::string ts = std::to_string(static_cast<long long>(now_t));
        std::string bdim = std::to_string(TILE_DIM) + "x" + std::to_string(TILE_DIM);
        std::string gdim = std::to_string(grid.x) + "x" + std::to_string(grid.y);

        writer.write_row({ts, "naive", std::to_string(M),
                          bdim, gdim, std::to_string(avg_naive)});

        writer.write_row({ts, "shared_mem", std::to_string(M),
                          bdim, gdim, std::to_string(avg_shared)});
    }

    cudaFree(d_a);
    cudaFree(d_c);

    return 0;
}
