// Small portability layer shared by the native DAC codec: files, aligned buffers, threading.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dacn {

namespace fs = std::filesystem;

// fopen that takes UTF-8 / native paths on every platform (wide API on Windows).
FILE * open_file(const fs::path & p, const char * mode);
bool read_file(const fs::path & p, std::vector<uint8_t> & out, std::string & err, size_t max_size = SIZE_MAX);

// 64-byte aligned float buffers
struct aligned_deleter { void operator()(float * p) const; };
using fbuf = std::unique_ptr<float, aligned_deleter>;
fbuf alloc_floats(size_t n);

// Worker threads used by the codec (OpenMP when available, otherwise GCD / std::thread).
void set_threads(int n);
int get_threads();
// Default: physical cores, leaving 4 free on big hybrid CPUs (low-power cores slow down static
// partitioning, and it keeps the media player / UI responsive). DAC_THREADS overrides.
int default_threads();
void parallel_for(int64_t n, const std::function<void(int64_t begin, int64_t end)> & fn);

struct error : std::runtime_error { using std::runtime_error::runtime_error; };

}  // namespace dacn
