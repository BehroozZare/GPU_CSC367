// Header-only binary PGM (P5) reader / writer.
// No external dependencies -- the P5 format is just:
//   "P5\n<width> <height>\n<maxval>\n" followed by raw bytes.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct PgmImage {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

inline bool read_pgm(const std::string& path, PgmImage& img) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "pgm_io: cannot open '%s'\n", path.c_str());
        return false;
    }

    char magic[3] = {};
    if (std::fscanf(f, "%2s", magic) != 1 ||
        magic[0] != 'P' || magic[1] != '5') {
        std::fprintf(stderr, "pgm_io: '%s' is not a binary PGM (P5)\n",
                     path.c_str());
        std::fclose(f);
        return false;
    }

    // Skip comments (lines starting with '#')
    int c = std::fgetc(f);
    while (c != EOF) {
        if (c == '#') {
            while (c != '\n' && c != EOF) c = std::fgetc(f);
        } else if (c > ' ') {
            std::ungetc(c, f);
            break;
        }
        c = std::fgetc(f);
    }

    int maxval = 0;
    if (std::fscanf(f, "%d %d %d", &img.width, &img.height, &maxval) != 3 ||
        img.width <= 0 || img.height <= 0 || maxval <= 0 || maxval > 255) {
        std::fprintf(stderr, "pgm_io: bad header in '%s'\n", path.c_str());
        std::fclose(f);
        return false;
    }

    // Consume the single whitespace byte after maxval
    std::fgetc(f);

    size_t npixels = (size_t)img.width * img.height;
    img.pixels.resize(npixels);
    size_t got = std::fread(img.pixels.data(), 1, npixels, f);
    std::fclose(f);

    if (got != npixels) {
        std::fprintf(stderr, "pgm_io: expected %zu bytes, got %zu in '%s'\n",
                     npixels, got, path.c_str());
        return false;
    }
    return true;
}

inline bool write_pgm(const std::string& path, int width, int height,
                       const uint8_t* data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "pgm_io: cannot create '%s'\n", path.c_str());
        return false;
    }
    std::fprintf(f, "P5\n%d %d\n255\n", width, height);
    std::fwrite(data, 1, (size_t)width * height, f);
    std::fclose(f);
    return true;
}
