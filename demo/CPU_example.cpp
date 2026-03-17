#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "csv_writer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int compute_value(int x, int lane_id, int tuning_parameter) {
    int acc = x;
    if (lane_id < tuning_parameter) {
        for (int i = 0; i < 100; i++)
            acc = std::sqrt(acc * 1.00001f + 1.0f);
    } else {
        for (int i = 0; i < 100; i++)
            acc = std::sqrt(acc * 0.99999f + 0.5f);
    }
    return acc;
}

void multiply_cpu(const std::vector<int>& input,
                  std::vector<int>& output,
                  int num_elements,
                  int tuning_parameter) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < num_elements; ++i) {
        const int lane_id = i % 32;
        output[i] = compute_value(input[i], lane_id, tuning_parameter);
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <tuning_parameter>\n", argv[0]);
        return 1;
    }

    int tuning_parameter = std::atoi(argv[1]);

    const int num_elements = 10000000;
    std::vector<int> h_data(num_elements);
    std::vector<int> h_output(num_elements);

    for (int i = 0; i < num_elements; ++i) {
        h_data[i] = i;
    }

    std::printf("Data size: %d MB\n", num_elements * static_cast<int>(sizeof(int)) / 1024 / 1024);

#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    std::printf("OpenMP: enabled (max threads: %d)\n", max_threads);
#else
    const int max_threads = 1;
    std::printf("OpenMP: disabled\n");
#endif

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        multiply_cpu(h_data, h_output, num_elements, tuning_parameter);
    }
    auto end = std::chrono::high_resolution_clock::now();

    const auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const long long avg_us = duration_us / 10;
    std::printf("Time taken: %lld microseconds\n", avg_us);

    CsvWriter writer("/home/behrooz/Desktop/CSC367/GPU/output/cpu_wrap.csv",
                     {"timestamp",
                      "tuning_parameter",
                      "num_elements",
                      "openmp_enabled",
                      "max_threads",
                      "avg_time_us"});

    if (writer.good()) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);

        writer.write_row({
            std::to_string(static_cast<long long>(now_time_t)),
            std::to_string(tuning_parameter),
            std::to_string(num_elements),
#ifdef _OPENMP
            "1",
#else
            "0",
#endif
            std::to_string(max_threads),
            std::to_string(avg_us),
        });
    }

    return 0;
}
