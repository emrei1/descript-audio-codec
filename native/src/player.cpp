#include "player.h"

#include "paths.h"

#include <cstdio>

namespace dacn {

namespace {

// FNV-1a 64 over the file contents: the same .dac always maps to the same cached WAV.
uint64_t file_hash(const fs::path & p) {
    uint64_t h = 1469598103934665603ull;
    FILE * f = open_file(p, "rb");
    if (!f) return h;
    unsigned char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0)
        for (size_t i = 0; i < n; ++i) { h ^= buf[i]; h *= 1099511628211ull; }
    fclose(f);
    return h;
}

void prune_cache(const fs::path & dir) {
    std::error_code ec;
    const auto now = fs::file_time_type::clock::now();
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        if (e.path().extension() != ".wav" && e.path().extension() != ".partial") continue;
        const auto t = e.last_write_time(ec);
        if (!ec && now - t > std::chrono::hours(24 * 7)) fs::remove(e.path(), ec);
    }
}

}  // namespace

fs::path cached_wav_for(const fs::path & input) {
    const fs::path dir = cache_dir();
    prune_cache(dir);
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long) file_hash(input));
    return dir / fs::u8path(input.stem().u8string() + "-" + hex + ".wav");
}

player_job::player_job(fs::path input, fs::path wav_out) : in_(std::move(input)), out_(std::move(wav_out)) {}

std::string player_job::title() const { return in_.stem().u8string(); }

void player_job::run() {
    model m;
    dac_file f;
    decompress_result r;
    if (!dac_file_load(in_, f, err_)) err_ = "Could not read the file: " + err_;
    else if (!m.load(default_model_path(), err_)) err_ = "Could not load the DAC model: " + err_;
    else {
        set_threads(default_threads());
        audio_seconds_ = (double) f.original_length / (double) f.sample_rate;
        engine eng(m);
        stage_ = 1;
        t_decode_ = std::chrono::steady_clock::now();
        if (decompress(eng, f, r, err_, decompress_options{}, &prog_)) {
            stage_ = 2;
            fs::path tmp = out_;
            tmp += ".partial";
            std::error_code ec;
            if (write_wav_f32(tmp, r.audio.samples.data(), r.audio.frames, r.audio.channels, r.audio.sample_rate, err_)) {
                fs::rename(tmp, out_, ec);
                if (ec) err_ = "Could not move the decoded file into the cache: " + ec.message();
                else ok_ = true;
            }
            if (!ok_) fs::remove(tmp, ec);
        }
    }
    stage_ = 3;
    finished_ = true;
}

double player_job::fraction() const {
    if (stage_ >= 2) return 1.0;
    const int64_t tot = prog_.total;
    return (stage_ == 1 && tot > 0) ? (double) prog_.done / (double) tot : 0.0;
}

std::string player_job::status() const {
    char buf[256];
    const int st = stage_;
    if (st == 0) return "Loading the DAC model...";
    if (st >= 2) return "Writing audio...";
    const double fr = fraction();
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_decode_).count();
    if (fr > 0.0) {
        const double eta = el / fr * (1.0 - fr);
        snprintf(buf, sizeof buf, "Decoding %.0f s of audio - %d%%, about %.0f s left", audio_seconds_, (int) (fr * 100), eta);
    } else {
        snprintf(buf, sizeof buf, "Decoding %.0f s of audio...", audio_seconds_);
    }
    return buf;
}

}  // namespace dacn
