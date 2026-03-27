#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#include "csv_writer.h"
#include "pgm_io.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Box blur kernel that works on both full images and horizontal strips.
//
//   in       -- input pixels (strip rows including top/bottom halo)
//   out      -- output pixels (strip rows only, no halo)
//   width    -- image width (same for input and output)
//   in_h     -- number of rows in the input buffer  (strip_h + top_halo + bot_halo)
//   out_h    -- number of rows in the output buffer  (strip_h)
//   top_halo -- how many halo rows precede the first output row in the input
//   R        -- blur radius (kernel window is (2R+1) x (2R+1))
__global__ void box_blur(const float* __restrict__ in,
                         float* __restrict__ out,
                         int width, int in_h, int out_h,
                         int top_halo, int R) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= out_h) return;

    int cy = y + top_halo;
    float sum = 0.0f;
    int count = 0;
    for (int dy = -R; dy <= R; dy++) {
        int ny = cy + dy;
        if (ny < 0 || ny >= in_h) continue;
        for (int dx = -R; dx <= R; dx++) {
            int nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            sum += in[ny * width + nx];
            count++;
        }
    }
    out[y * width + x] = sum / count;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <input.pgm> [num_streams] [blur_radius]\n", argv[0]);
        return 1;
    }

    const char* pgm_path = argv[1];
    int num_streams = (argc >= 3) ? std::atoi(argv[2]) : 4;
    int R           = (argc >= 4) ? std::atoi(argv[3]) : 7;

    // ---- Load image ----
    PgmImage img;
    if (!read_pgm(pgm_path, img)) return 1;

    int width  = img.width;
    int height = img.height;
    size_t npixels = (size_t)width * height;

    printf("=== CUDA Stream Image Blur Demo ===\n");
    printf("Image        : %s (%d x %d = %.1f MB)\n",
           pgm_path, width, height, npixels * sizeof(float) / (1024.0 * 1024.0));
    printf("Blur radius  : %d  (window %dx%d)\n", R, 2*R+1, 2*R+1);
    printf("Streams      : %d\n\n", num_streams);

    // ---- Pinned host memory (required for truly async cudaMemcpyAsync) ----
    float *h_input = nullptr, *h_output_serial = nullptr, *h_output_pipe = nullptr;
    GPU_CHECK(cudaMallocHost(&h_input,         npixels * sizeof(float)));
    GPU_CHECK(cudaMallocHost(&h_output_serial, npixels * sizeof(float)));
    GPU_CHECK(cudaMallocHost(&h_output_pipe,   npixels * sizeof(float)));

    for (size_t i = 0; i < npixels; i++)
        h_input[i] = (float)img.pixels[i];

    // ---- Device memory for serial path ----
    float *d_in = nullptr, *d_out = nullptr;
    GPU_CHECK(cudaMalloc(&d_in,  npixels * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_out, npixels * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid_full((width + block.x - 1) / block.x,
                   (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    GPU_CHECK(cudaEventCreate(&start));
    GPU_CHECK(cudaEventCreate(&stop));

    // ---- Warm-up ----
    GPU_CHECK(cudaMemcpy(d_in, h_input, npixels * sizeof(float), cudaMemcpyHostToDevice));
    box_blur<<<grid_full, block>>>(d_in, d_out, width, height, height, 0, R);
    GPU_CHECK(cudaDeviceSynchronize());

    // ================================================================
    //  SERIAL: H2D -> kernel -> D2H  (all on default stream)
    // ================================================================
    GPU_CHECK(cudaEventRecord(start));

    GPU_CHECK(cudaMemcpy(d_in, h_input, npixels * sizeof(float), cudaMemcpyHostToDevice));
    box_blur<<<grid_full, block>>>(d_in, d_out, width, height, height, 0, R);
    GPU_CHECK(cudaMemcpy(h_output_serial, d_out, npixels * sizeof(float), cudaMemcpyDeviceToHost));

    GPU_CHECK(cudaEventRecord(stop));
    GPU_CHECK(cudaEventSynchronize(stop));

    float serial_ms = 0.0f;
    GPU_CHECK(cudaEventElapsedTime(&serial_ms, start, stop));

    // ================================================================
    //  PIPELINED: split image into strips, overlap H2D / kernel / D2H
    // ================================================================
    std::vector<cudaStream_t> streams(num_streams);
    for (int s = 0; s < num_streams; s++)
        GPU_CHECK(cudaStreamCreate(&streams[s]));

    int base_strip_h = height / num_streams;

    // Find the largest input chunk size across all strips to size device buffers
    int max_in_rows = 0;
    for (int s = 0; s < num_streams; s++) {
        int strip_start = s * base_strip_h;
        int strip_h = (s == num_streams - 1) ? height - strip_start : base_strip_h;
        int top_halo = std::min(R, strip_start);
        int bot_halo = std::min(R, height - strip_start - strip_h);
        int in_rows  = strip_h + top_halo + bot_halo;
        if (in_rows > max_in_rows) max_in_rows = in_rows;
    }

    // Per-stream device buffers
    std::vector<float*> d_in_chunk(num_streams), d_out_chunk(num_streams);
    for (int s = 0; s < num_streams; s++) {
        GPU_CHECK(cudaMalloc(&d_in_chunk[s],  (size_t)max_in_rows * width * sizeof(float)));
        GPU_CHECK(cudaMalloc(&d_out_chunk[s], (size_t)base_strip_h * width * sizeof(float)));
    }

    GPU_CHECK(cudaDeviceSynchronize());
    GPU_CHECK(cudaEventRecord(start));

    for (int s = 0; s < num_streams; s++) {
        int strip_start = s * base_strip_h;
        int strip_h = (s == num_streams - 1) ? height - strip_start : base_strip_h;
        int top_halo = std::min(R, strip_start);
        int bot_halo = std::min(R, height - strip_start - strip_h);
        int in_start = strip_start - top_halo;
        int in_rows  = strip_h + top_halo + bot_halo;

        // H2D: copy input strip (with halo) to per-stream device buffer
        GPU_CHECK(cudaMemcpyAsync(
            d_in_chunk[s],
            h_input + (size_t)in_start * width,
            (size_t)in_rows * width * sizeof(float),
            cudaMemcpyHostToDevice,
            streams[s]));

        // Kernel: blur the strip
        dim3 grid_strip((width + block.x - 1) / block.x,
                        (strip_h + block.y - 1) / block.y);
        box_blur<<<grid_strip, block, 0, streams[s]>>>(
            d_in_chunk[s], d_out_chunk[s],
            width, in_rows, strip_h, top_halo, R);

        // D2H: copy output strip back
        GPU_CHECK(cudaMemcpyAsync(
            h_output_pipe + (size_t)strip_start * width,
            d_out_chunk[s],
            (size_t)strip_h * width * sizeof(float),
            cudaMemcpyDeviceToHost,
            streams[s]));
    }

    GPU_CHECK(cudaEventRecord(stop));
    GPU_CHECK(cudaEventSynchronize(stop));

    float pipelined_ms = 0.0f;
    GPU_CHECK(cudaEventElapsedTime(&pipelined_ms, start, stop));

    float speedup = serial_ms / pipelined_ms;

    printf("Serial     time : %.3f ms  (H2D + blur + D2H)\n", serial_ms);
    printf("Pipelined  time : %.3f ms  (%d streams)\n", pipelined_ms, num_streams);
    printf("Speedup         : %.2fx\n", speedup);

    // ---- Correctness check ----
    float max_diff = 0.0f;
    for (size_t i = 0; i < npixels; i++) {
        float diff = std::fabs(h_output_serial[i] - h_output_pipe[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff        : %e\n", max_diff);

    // ---- Save blurred image ----
    std::vector<uint8_t> out_bytes(npixels);
    for (size_t i = 0; i < npixels; i++)
        out_bytes[i] = (uint8_t)std::min(255.0f, std::max(0.0f, h_output_serial[i]));
    write_pgm("/home/behrooz/Desktop/CSC367/GPU/output/blurred.pgm",
              width, height, out_bytes.data());

    // ---- CSV logging ----
    CsvWriter writer("/home/behrooz/Desktop/CSC367/GPU/output/stream_image_blur.csv",
                     {"timestamp",
                      "width",
                      "height",
                      "blur_radius",
                      "num_streams",
                      "serial_time_ms",
                      "pipelined_time_ms",
                      "speedup"});

    if (writer.good()) {
        auto now = std::chrono::system_clock::now();
        auto now_t = std::chrono::system_clock::to_time_t(now);

        char s_buf[32], p_buf[32], sp_buf[32];
        std::snprintf(s_buf,  sizeof(s_buf),  "%.3f", serial_ms);
        std::snprintf(p_buf,  sizeof(p_buf),  "%.3f", pipelined_ms);
        std::snprintf(sp_buf, sizeof(sp_buf), "%.2f", speedup);

        writer.write_row({
            std::to_string(static_cast<long long>(now_t)),
            std::to_string(width),
            std::to_string(height),
            std::to_string(R),
            std::to_string(num_streams),
            s_buf,
            p_buf,
            sp_buf
        });
    }

    // ---- Cleanup ----
    for (int s = 0; s < num_streams; s++) {
        cudaFree(d_in_chunk[s]);
        cudaFree(d_out_chunk[s]);
        cudaStreamDestroy(streams[s]);
    }
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output_serial);
    cudaFreeHost(h_output_pipe);

    return 0;
}
