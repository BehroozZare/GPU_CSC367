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

__global__ void strided_copy_kernel(const int* __restrict__ d_input,
                                    int* __restrict__ d_output,
                                    int num_output_elements,
                                    int stride){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_output_elements) return;

    d_output[tid] = d_input[tid * stride];
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <stride> <threads_per_block>\n", argv[0]);
        return 1;
    }

    int stride = std::atoi(argv[1]);
    int thread_per_block = std::atoi(argv[2]);

    const int X = 1000000;
    const int input_size = 32 * X;
    const int output_size = X;

    if (stride < 1 || stride > 32) {
        std::fprintf(stderr, "stride must be between 1 and 32\n");
        return 1;
    }

    std::vector<int> h_input(input_size);
    std::vector<int> h_output(output_size);
    int* d_input  = nullptr;
    int* d_output = nullptr;

    for (int i = 0; i < input_size; i++){
        h_input[i] = i;
    }

    printf("Input size : %lu MB (%d elements)\n",
           (unsigned long)(input_size * sizeof(int)) / 1024 / 1024, input_size);
    printf("Output size: %lu MB (%d elements)\n",
           (unsigned long)(output_size * sizeof(int)) / 1024 / 1024, output_size);
    printf("Stride     : %d\n", stride);

    GPU_CHECK(cudaMalloc(&d_input,  input_size  * sizeof(int)));
    GPU_CHECK(cudaMalloc(&d_output, output_size * sizeof(int)));
    GPU_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_size(thread_per_block);
    dim3 grid_size((output_size + thread_per_block - 1) / thread_per_block);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++){
        strided_copy_kernel<<<grid_size, block_size>>>(d_input, d_output, output_size, stride);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();

    GPU_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(int), cudaMemcpyDeviceToHost));
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    long long avg_us = duration / 10;
    printf("Time taken: %lld microseconds\n", avg_us);

    CsvWriter writer("/home/behrooz/Desktop/CSC367/GPU/output/warp_stride.csv",
                     {"timestamp",
                      "stride",
                      "X",
                      "input_size",
                      "block_size",
                      "grid_size",
                      "avg_time_us"});

    if (writer.good()) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);

        writer.write_row({
            std::to_string(static_cast<long long>(now_time_t)),
            std::to_string(stride),
            std::to_string(X),
            std::to_string(input_size),
            std::to_string(thread_per_block),
            std::to_string(grid_size.x),
            std::to_string(avg_us)
        });
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
