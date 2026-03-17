// Simple CSV writer utility for host-side logging.
// Header-only and minimal so it can be used from CUDA .cu files.

#pragma once

#include <fstream>
#include <string>
#include <vector>

class CsvWriter {
public:
    CsvWriter(const std::string& path,
              const std::vector<std::string>& header)
        : path_(path)
    {
        // Check if the file exists and is non-empty so we know
        // whether to emit the header.
        bool write_header = false;
        {
            std::ifstream in(path_);
            if (!in.good()) {
                write_header = true;
            } else if (in.peek() == std::ifstream::traits_type::eof()) {
                write_header = true;
            }
        }

        file_.open(path_, std::ios::app);
        if (!file_.is_open()) {
            valid_ = false;
            return;
        }

        valid_ = true;
        if (write_header) {
            write_row(header);
        }
    }

    bool good() const { return valid_; }

    // Append a single CSV row (no quoting/escaping).
    void write_row(const std::vector<std::string>& columns)
    {
        if (!valid_) return;

        bool first = true;
        for (const auto& col : columns) {
            if (!first) {
                file_ << ',';
            }
            first = false;
            file_ << col;
        }
        file_ << '\n';
        file_.flush();
    }

private:
    std::string   path_;
    std::ofstream file_;
    bool          valid_ = false;
};

