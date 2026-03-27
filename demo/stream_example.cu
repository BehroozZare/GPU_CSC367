#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
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

// Compute-heavy kernel A.
// Launched with (numSMs/2) blocks × 768 threads = 24 warps/SM on half the SMs.
// Serial: only half the SMs are busy.  Concurrent with kernel_B on a
// separate stream: all SMs are busy → ~2× speedup.
__global__ __launch_bounds__(768)
void kernel_A(float* __restrict__ out, int N, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        float val = (float)i;
        for (int j = 0; j < iterations; j++)
            val = val * 0.9999f + 0.0001f;
        out[i] = val;
    }
}

// Compute-heavy kernel B: same structure, different constants so the
// compiler treats it as a distinct kernel.
__global__ __launch_bounds__(768)
void kernel_B(float* __restrict__ out, int N, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        float val = (float)(N - i);
        for (int j = 0; j < iterations; j++)
            val = val * 0.9998f + 0.0002f;
        out[i] = val;
    }
}

int main(int argc, char* argv[]) {
    int iterations = 5000;
    if (argc >= 2)
        iterations = std::atoi(argv[1]);

    int N = 1 << 22;

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;

    int threads_per_block = 768;
    int warps_per_sm = threads_per_block / 32;
    int blocks_per_kernel = num_sms / 2;

    printf("=== CUDA Stream Concurrency Demo ===\n");
    printf("Device       : %s (%d SMs)\n", prop.name, num_sms);
    printf("Block size   : %d threads  (%d warps = 50%% occupancy per SM)\n",
           threads_per_block, warps_per_sm);
    printf("Grid size    : %d blocks per kernel  (half the SMs)\n", blocks_per_kernel);
    printf("  Serial     : each kernel uses %d/%d SMs in turn\n", blocks_per_kernel, num_sms);
    printf("  Concurrent : both kernels together fill all %d SMs\n", num_sms);
    printf("Iterations   : %d\n", iterations);
    printf("N            : %d elements\n\n", N);

    float *d_out_A = nullptr, *d_out_B = nullptr;
    GPU_CHECK(cudaMalloc(&d_out_A, N * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_out_B, N * sizeof(float)));

    dim3 block(threads_per_block);
    dim3 grid(blocks_per_kernel);

    cudaEvent_t start, stop;
    GPU_CHECK(cudaEventCreate(&start));
    GPU_CHECK(cudaEventCreate(&stop));

    // Warm-up: run each kernel once to avoid first-launch overhead
    kernel_A<<<grid, block>>>(d_out_A, N, 1);
    kernel_B<<<grid, block>>>(d_out_B, N, 1);
    GPU_CHECK(cudaDeviceSynchronize());

    // ---- Serial execution (default stream) ----
    nvtxRangePushA("Serial");
    GPU_CHECK(cudaEventRecord(start));
    kernel_A<<<grid, block>>>(d_out_A, N, iterations);
    kernel_B<<<grid, block>>>(d_out_B, N, iterations);
    GPU_CHECK(cudaEventRecord(stop));
    GPU_CHECK(cudaEventSynchronize(stop));
    nvtxRangePop();

    float serial_ms = 0.0f;
    GPU_CHECK(cudaEventElapsedTime(&serial_ms, start, stop));

    // ---- Concurrent execution (two streams) ----
    cudaStream_t stream1, stream2;
    GPU_CHECK(cudaStreamCreate(&stream1));
    GPU_CHECK(cudaStreamCreate(&stream2));

    nvtxRangePushA("Concurrent");
    GPU_CHECK(cudaEventRecord(start));
    kernel_A<<<grid, block, 0, stream1>>>(d_out_A, N, iterations);
    kernel_B<<<grid, block, 0, stream2>>>(d_out_B, N, iterations);
    GPU_CHECK(cudaEventRecord(stop));
    GPU_CHECK(cudaEventSynchronize(stop));
    nvtxRangePop();

    float concurrent_ms = 0.0f;
    GPU_CHECK(cudaEventElapsedTime(&concurrent_ms, start, stop));

    float speedup = serial_ms / concurrent_ms;

    printf("Serial     time : %.3f ms\n", serial_ms);
    printf("Concurrent time : %.3f ms\n", concurrent_ms);
    printf("Speedup         : %.2fx\n", speedup);

    // ---- CSV logging ----
    CsvWriter writer("/home/behrooz/Desktop/CSC367/GPU/output/stream_concurrency.csv",
                     {"timestamp",
                      "iterations",
                      "num_elements",
                      "block_size",
                      "grid_size",
                      "serial_time_ms",
                      "concurrent_time_ms",
                      "speedup"});

    if (writer.good()) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);

        char serial_buf[32], concurrent_buf[32], speedup_buf[32];
        std::snprintf(serial_buf, sizeof(serial_buf), "%.3f", serial_ms);
        std::snprintf(concurrent_buf, sizeof(concurrent_buf), "%.3f", concurrent_ms);
        std::snprintf(speedup_buf, sizeof(speedup_buf), "%.2f", speedup);

        writer.write_row({
            std::to_string(static_cast<long long>(now_time_t)),
            std::to_string(iterations),
            std::to_string(N),
            std::to_string(threads_per_block),
            std::to_string(grid.x),
            serial_buf,
            concurrent_buf,
            speedup_buf
        });
    }

    GPU_CHECK(cudaStreamDestroy(stream1));
    GPU_CHECK(cudaStreamDestroy(stream2));
    GPU_CHECK(cudaEventDestroy(start));
    GPU_CHECK(cudaEventDestroy(stop));
    cudaFree(d_out_A);
    cudaFree(d_out_B);

    return 0;
}
