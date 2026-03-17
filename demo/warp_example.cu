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

__global__ void divergence_kernel(int* d_input, int* d_output, int num_elements, int tuning_parameter){
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;

    if (global_tid >= num_elements) {
        return;
    }

    int acc = d_input[global_tid];
    if (lane_id < tuning_parameter) {
        for (int i = 0; i < 100; i++)
            acc = sqrtf(acc * 1.00001f + 1.0f);
    } else {
        for (int i = 0; i < 100; i++)
            acc = sqrtf(acc * 0.99999f + 0.5f);
    }
    d_output[global_tid] = acc;
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <tuning_parameter> <threads_per_block>\n", argv[0]);
        return 1;
    }

    int tuning_parameter = atoi(argv[1]);
    int thread_per_block = std::atoi(argv[2]);

    int num_elements = 1e7;
    std::vector<int> h_data(num_elements);
    std::vector<int> h_output(num_elements);
    int* d_data = nullptr;
    int* d_output = nullptr;

    for(int i = 0; i < num_elements; i++){
        h_data[i] = i;
    }

    printf("Data size: %lu MB\n", num_elements * sizeof(int) / 1024 / 1024);
    GPU_CHECK(cudaMalloc(&d_data, num_elements * sizeof(int)));
    GPU_CHECK(cudaMalloc(&d_output, num_elements * sizeof(int)));
    GPU_CHECK(cudaMemcpy(d_data, h_data.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_size(thread_per_block);
    dim3 grid_size((num_elements + thread_per_block - 1) / thread_per_block);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++){
        divergence_kernel<<<grid_size, block_size>>>(d_data, d_output, num_elements, tuning_parameter);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();

    GPU_CHECK(cudaMemcpy(h_output.data(), d_output, num_elements * sizeof(int), cudaMemcpyDeviceToHost));
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    long long avg_us = duration / 10;
    printf("Time taken: %lld microseconds\n", avg_us);

    CsvWriter writer("/home/behrooz/Desktop/CSC367/GPU/output/warp_divergence.csv",
                     {"timestamp",
                      "tuning_parameter",
                      "num_elements",
                      "block_size",
                      "grid_size",
                      "avg_time_us"});

    if (writer.good()) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);

        writer.write_row({
            std::to_string(static_cast<long long>(now_time_t)),
            std::to_string(tuning_parameter),
            std::to_string(num_elements),
            std::to_string(thread_per_block),
            std::to_string(grid_size.x),
            std::to_string(avg_us)
        });
    }

    cudaFree(d_data);
    cudaFree(d_output);

    return 0;
}
