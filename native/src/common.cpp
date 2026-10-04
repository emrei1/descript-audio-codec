#include "common.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <malloc.h>
#else
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#elif defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

namespace dacn {

FILE * open_file(const fs::path & p, const char * mode) {
#ifdef _WIN32
    std::wstring m(mode, mode + strlen(mode));
    return _wfopen(p.c_str(), m.c_str());
#else
    return fopen(p.c_str(), mode);
#endif
}

bool read_file(const fs::path & p, std::vector<uint8_t> & out, std::string & err, size_t max_size) {
    FILE * f = open_file(p, "rb");
    if (!f) { err = "cannot open " + p.u8string(); return false; }
    std::vector<uint8_t> buf;
    uint8_t chunk[1 << 16];
    size_t n;
    while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) {
        if (buf.size() + n > max_size) { fclose(f); err = "file too large"; return false; }
        buf.insert(buf.end(), chunk, chunk + n);
    }
    const bool bad = ferror(f) != 0;
    fclose(f);
    if (bad) { err = "read error on " + p.u8string(); return false; }
    out.swap(buf);
    return true;
}

void aligned_deleter::operator()(float * p) const {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

fbuf alloc_floats(size_t n) {
    const size_t bytes = std::max<size_t>(64, (n * sizeof(float) + 63) / 64 * 64);
#ifdef _WIN32
    void * p = _aligned_malloc(bytes, 64);
#else
    void * p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) p = nullptr;
#endif
    if (!p) throw std::bad_alloc();
    return fbuf((float *) p);
}

static std::atomic<int> g_threads{0};

void set_threads(int n) {
    n = std::max(1, n);
    g_threads = n;
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
}

int get_threads() {
    const int n = g_threads.load();
    return n > 0 ? n : default_threads();
}

static int physical_cores() {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<uint8_t> buf(len);
    int cores = 0;
    if (len && GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        for (DWORD off = 0; off < len;) {
            auto * p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
            if (p->Relationship == RelationProcessorCore) ++cores;
            off += p->Size;
        }
    }
    if (cores > 0) return cores;
#endif
    const unsigned hc = std::thread::hardware_concurrency();
    return hc ? (int) hc : 4;
}

int default_threads() {
    if (const char * e = getenv("DAC_THREADS")) {
        const int n = atoi(e);
        if (n > 0) return n;
    }
    const int c = physical_cores();
    return c >= 8 ? c - 4 : c;
}

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)> & fn) {
    if (n <= 0) return;
    const int nt = (int) std::min<int64_t>(get_threads(), n);
    if (nt <= 1) { fn(0, n); return; }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int i = 0; i < nt; ++i) fn(n * i / nt, n * (i + 1) / nt);
#elif defined(__APPLE__)
    dispatch_apply((size_t) nt, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t i) { fn(n * (int64_t) i / nt, n * (int64_t) (i + 1) / nt); });
#else
    std::vector<std::thread> th;
    for (int i = 1; i < nt; ++i) th.emplace_back([&, i] { fn(n * i / nt, n * (i + 1) / nt); });
    fn(0, n / nt);
    for (auto & t : th) t.join();
#endif
}

}  // namespace dacn
