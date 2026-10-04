#include "paths.h"

#include <cstdlib>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shlobj.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

namespace dacn {

fs::path executable_path() {
#ifdef _WIN32
    wchar_t p[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameW(nullptr, p, (DWORD) (sizeof p / sizeof p[0]));
    return fs::path(std::wstring(p, n));
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    _NSGetExecutablePath(buf.data(), &size);
    std::error_code ec;
    const fs::path p = fs::canonical(fs::path(buf.c_str()), ec);
    return ec ? fs::path(buf.c_str()) : p;
#else
    std::error_code ec;
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::path() : p;
#endif
}

static fs::path home() {
    const char * h = getenv("HOME");
    return h ? fs::path(h) : fs::path(".");
}

fs::path default_model_path() {
    if (const char * e = getenv("DAC_MODEL")) return fs::u8path(e);
    const fs::path dir = executable_path().parent_path();
    std::error_code ec;
    if (fs::exists(dir / "dac.gguf", ec)) return dir / "dac.gguf";
#ifdef __APPLE__
    if (fs::exists(dir / ".." / "Resources" / "dac.gguf", ec)) return dir / ".." / "Resources" / "dac.gguf";
    return home() / "Library" / "Application Support" / "DAC Player" / "dac.gguf";
#elif defined(_WIN32)
    return dir / "dac.gguf";
#else
    const char * x = getenv("XDG_DATA_HOME");
    return (x && *x ? fs::path(x) : home() / ".local" / "share") / "dac-player" / "dac.gguf";
#endif
}

fs::path cache_dir() {
    fs::path d;
#ifdef _WIN32
    PWSTR base = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &base))) d = fs::path(base);
    CoTaskMemFree(base);
    d = d / "DacPlayer" / "cache";
#elif defined(__APPLE__)
    d = home() / "Library" / "Caches" / "com.descript.dac-player";
#else
    const char * x = getenv("XDG_CACHE_HOME");
    d = (x && *x ? fs::path(x) : home() / ".cache") / "dac-player";
#endif
    std::error_code ec;
    fs::create_directories(d, ec);
    return d;
}

}  // namespace dacn
