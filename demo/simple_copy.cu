#include <cuda_runtime.h>
#include <cuda/std/cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
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

__global__ void simpleCopyKernel(unsigned long long loopCount, uint4 *dst, uint4 *src) {
    for (unsigned int i = 0; i < loopCount; i++) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t offset = idx * sizeof(uint4);
        uint4* dst_uint4 = reinterpret_cast<uint4*>((char*)dst + offset);
        uint4* src_uint4 = reinterpret_cast<uint4*>((char*)src + offset);
        __stcg(dst_uint4, __ldcg(src_uint4));
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <threads_per_block>\n", argv[0]);
        return 1;
    }

    int thread_per_block = std::atoi(argv[1]);

    const int X = 1000000;
    // 1M ints = 4MB; pack into uint4 (16 bytes each) -> 250K uint4 elements
    const int num_uint4 = (X * sizeof(int) + sizeof(uint4) - 1) / sizeof(uint4);
    const size_t total_bytes = (size_t)num_uint4 * sizeof(uint4);

    printf("Copy size  : %zu bytes (%d uint4 elements)\n", total_bytes, num_uint4);
    printf("Block size : %d\n", thread_per_block);

    std::vector<unsigned char> h_src(total_bytes);
    std::vector<unsigned char> h_dst(total_bytes, 0);
    for (size_t i = 0; i < total_bytes; i++) {
        h_src[i] = (unsigned char)(i & 0xFF);
    }

    uint4 *d_src = nullptr, *d_dst = nullptr;
    GPU_CHECK(cudaMalloc(&d_src, total_bytes));
    GPU_CHECK(cudaMalloc(&d_dst, total_bytes));
    GPU_CHECK(cudaMemcpy(d_src, h_src.data(), total_bytes, cudaMemcpyHostToDevice));

    dim3 block_size(thread_per_block);
    dim3 grid_size((num_uint4 + thread_per_block - 1) / thread_per_block);

    printf("Grid size  : %d\n", grid_size.x);

    // Warmup
    simpleCopyKernel<<<grid_size, block_size>>>(1, d_dst, d_src);
    GPU_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        simpleCopyKernel<<<grid_size, block_size>>>(1, d_dst, d_src);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();

    GPU_CHECK(cudaMemcpy(h_dst.data(), d_dst, total_bytes, cudaMemcpyDeviceToHost));
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    long long avg_us = duration / 100;
    printf("Time taken : %lld microseconds\n", avg_us);

    CsvWriter writer("/home/behrooz/Desktop/CSC367/GPU/output/simply_copy.csv",
                     {"timestamp",
                      "X",
                      "block_size",
                      "grid_size",
                      "avg_time_us"});

    if (writer.good()) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);

        writer.write_row({
            std::to_string(static_cast<long long>(now_time_t)),
            std::to_string(X),
            std::to_string(thread_per_block),
            std::to_string(grid_size.x),
            std::to_string(avg_us)
        });
    }

    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
