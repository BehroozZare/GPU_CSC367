#include <cuda_runtime.h>
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

// Pointer-chasing kernel: each thread follows a chain of dependent global
// memory reads.  Every load address depends on the *value* returned by the
// previous load, so the loads are strictly serial within a thread.  The only
// way the hardware can keep the memory pipeline busy is by switching to other
// warps while one is stalled -- i.e. latency hiding.
//
// We launch exactly one block per SM so that threads_per_block directly
// controls the number of resident warps.  Each thread processes multiple
// elements via a grid-stride loop to keep total work constant.
__global__ void latency_hiding_kernel(const int* __restrict__ lookup,
                                      int* __restrict__ output,
                                      int num_elements,
                                      int chain_length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int elem = tid; elem < num_elements; elem += total_threads) {
        int idx = elem;
        for (int i = 0; i < chain_length; i++)
            idx = lookup[idx];
        output[elem] = idx;
    }
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <chain_length> <threads_per_block>\n", argv[0]);
        return 1;
    }

    int chain_length    = std::atoi(argv[1]);
    int thread_per_block = std::atoi(argv[2]);

    int num_elements = 1e7;
    std::vector<int> h_lookup(num_elements);
    std::vector<int> h_output(num_elements);
    int* d_lookup = nullptr;
    int* d_output = nullptr;

    std::srand(42);
    for (int i = 0; i < num_elements; i++) {
        h_lookup[i] = std::rand() % num_elements;
    }

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;

    printf("Lookup table : %lu MB (%d elements)\n",
           (unsigned long)(num_elements * sizeof(int)) / 1024 / 1024, num_elements);
    printf("Chain length : %d\n", chain_length);
    printf("Block size   : %d  (= %d warps per SM)\n", thread_per_block, thread_per_block / 32);
    printf("Grid size    : %d  (one block per SM)\n", num_sms);

    GPU_CHECK(cudaMalloc(&d_lookup, num_elements * sizeof(int)));
    GPU_CHECK(cudaMalloc(&d_output, num_elements * sizeof(int)));
    GPU_CHECK(cudaMemcpy(d_lookup, h_lookup.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_size(thread_per_block);
    dim3 grid_size(num_sms);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        latency_hiding_kernel<<<grid_size, block_size>>>(d_lookup, d_output, num_elements, chain_length);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();

    GPU_CHECK(cudaMemcpy(h_output.data(), d_output, num_elements * sizeof(int), cudaMemcpyDeviceToHost));
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    long long avg_us = duration / 10;
    printf("Time taken: %lld microseconds\n", avg_us);

    CsvWriter writer("/home/behrooz/Desktop/CSC367/GPU/output/latency_hiding.csv",
                     {"timestamp",
                      "chain_length",
                      "num_elements",
                      "block_size",
                      "grid_size",
                      "avg_time_us"});

    if (writer.good()) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);

        writer.write_row({
            std::to_string(static_cast<long long>(now_time_t)),
            std::to_string(chain_length),
            std::to_string(num_elements),
            std::to_string(thread_per_block),
            std::to_string(grid_size.x),
            std::to_string(avg_us)
        });
    }

    cudaFree(d_lookup);
    cudaFree(d_output);

    return 0;
}
