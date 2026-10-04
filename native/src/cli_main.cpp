// dac-native: command-line front end of the native CPU codec.
//
//   dac-native encode <in.wav|flac|mp3> <out.dac> [--win 5.0] [--n_quantizers N]
//   dac-native decode <in.dac> <out.wav> [--reference]
//   dac-native info   <in.dac>
//   dac-native bench  [--frames 416]
// common options: --model dac.gguf  --threads N
#include "codec.h"
#include "paths.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace dacn;

namespace {

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int usage() {
    fprintf(stderr,
            "usage:\n"
            "  dac-native encode <in.wav|flac|mp3> <out.dac> [--win SECONDS] [--n_quantizers N]\n"
            "  dac-native decode <in.dac> <out.wav> [--reference]\n"
            "  dac-native info   <in.dac>\n"
            "  dac-native bench  [--frames N]\n"
            "options: --model PATH (default: next to the executable or $DAC_MODEL)  --threads N\n");
    return 2;
}

bool load_model(model & m, const fs::path & p) {
    std::string err;
    const double t = now();
    if (!m.load(p, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return false; }
    fprintf(stderr, "model: %s (%.0f ms)\n", p.u8string().c_str(), (now() - t) * 1e3);
    return true;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t ** wargv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) args.push_back(fs::path(wargv[i]).u8string());
#else
int main(int argc, char ** argv) {
    std::vector<std::string> args(argv, argv + argc);
#endif
    if (args.size() < 2) return usage();
    const std::string cmd = args[1];
    std::vector<std::string> pos;
    fs::path model_path = default_model_path();
    int threads = default_threads(), frames = 416, n_q = 0;
    double win = 5.0;
    bool reference = false;
    for (size_t i = 2; i < args.size(); ++i) {
        const std::string & a = args[i];
        auto next = [&]() -> std::string { if (i + 1 >= args.size()) { usage(); exit(2); } return args[++i]; };
        if (a == "--model") model_path = fs::u8path(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--frames") frames = std::stoi(next());
        else if (a == "--win") win = std::stod(next());
        else if (a == "--n_quantizers") n_q = std::stoi(next());
        else if (a == "--reference") reference = true;
        else pos.push_back(a);
    }
    set_threads(threads);
    std::string err;

    if (cmd == "info" && pos.size() == 1) {
        dac_file f;
        if (!dac_file_load(fs::u8path(pos[0]), f, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        printf("channels=%lld codebooks=%lld frames=%lld chunk_length=%lld padding=%s\n", (long long) f.channels,
               (long long) f.n_codebooks, (long long) f.T, (long long) f.chunk_length, f.padding ? "true" : "false");
        printf("sample_rate=%lld original_length=%lld (%.2f s) input_db=%.2f LUFS version=%s\n", (long long) f.sample_rate,
               (long long) f.original_length, (double) f.original_length / f.sample_rate, f.input_db, f.dac_version.c_str());
        return 0;
    }

    model m;
    if (!load_model(m, model_path)) return 1;
    engine eng(m);
    fprintf(stderr, "backend: %s, %d threads\n", eng.backend_name(), get_threads());

    if (cmd == "encode" && pos.size() == 2) {
        audio_buffer in;
        double t = now();
        if (!read_audio(fs::u8path(pos[0]), in, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        fprintf(stderr, "input: %.2f s, %d Hz, %d ch (read %.0f ms)\n", (double) in.frames / in.sample_rate,
                in.sample_rate, in.channels, (now() - t) * 1e3);
        compress_options opt;
        opt.win_duration = win;
        opt.n_quantizers = n_q;
        dac_file f;
        t = now();
        if (!compress(eng, in, f, err, opt)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        const double el = now() - t, dur = (double) in.frames / in.sample_rate;
        if (!dac_file_save(fs::u8path(pos[1]), f, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        printf("encoded %.2f s in %.2f s (%.1fx realtime): %lld ch x %lld codebooks x %lld frames -> %s\n", dur, el,
               dur / el, (long long) f.channels, (long long) f.n_codebooks, (long long) f.T, pos[1].c_str());
        return 0;
    }
    if (cmd == "decode" && pos.size() == 2) {
        dac_file f;
        if (!dac_file_load(fs::u8path(pos[0]), f, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        decompress_options opt;
        opt.reference = reference;
        decompress_result r;
        const double t = now();
        if (!decompress(eng, f, r, err, opt)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        const double el = now() - t, dur = (double) r.audio.frames / r.audio.sample_rate;
        if (!write_wav_f32(fs::u8path(pos[1]), r.audio.samples.data(), r.audio.frames, r.audio.channels,
                           r.audio.sample_rate, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        printf("decoded %.2f s (%d ch, %d Hz) in %.2f s (%.1fx realtime), network %.2f s, limiter %.2f dB, mode=%s -> %s\n",
               dur, r.audio.channels, r.audio.sample_rate, el, dur / el, r.neural_seconds, r.limiter_db,
               reference ? "reference" : "player", pos[1].c_str());
        return 0;
    }
    if (cmd == "bench") {
        // random codes; alternate padded / unpadded decodes plus one encode of the same length
        std::vector<std::vector<int32_t>> codes((size_t) m.n_codebooks, std::vector<int32_t>((size_t) frames));
        uint32_t s = 12345;
        for (auto & row : codes) for (auto & c : row) { s = s * 1664525u + 1013904223u; c = (int32_t) ((s >> 8) % m.codebook_size); }
        std::vector<const int32_t *> rows;
        for (auto & row : codes) rows.push_back(row.data());
        std::vector<float> out;
        for (int pad = 0; pad < 2; ++pad) {
            eng.decode(rows.data(), (int) rows.size(), frames, pad, out);   // warm-up (JIT, weight packing)
            double best = 1e30;
            for (int r = 0; r < 3; ++r) { const double t = now(); eng.decode(rows.data(), (int) rows.size(), frames, pad, out); best = std::min(best, now() - t); }
            const double sec = (double) out.size() / m.sample_rate;
            printf("decode %-8s %d frames -> %.2f s audio: %.3f s (RTF %.3f, %.1fx realtime)\n", pad ? "padded" : "unpadded",
                   frames, sec, best, best / sec, sec / best);
        }
        if (m.has_encoder()) {
            std::vector<float> x((size_t) frames * m.hop_length);
            for (size_t i = 0; i < x.size(); ++i) x[i] = 0.3f * std::sin(0.01f * (float) i) * std::sin(0.0003f * (float) i);
            std::vector<int32_t> c;
            int64_t fr = 0;
            eng.encode(x.data(), (int64_t) x.size(), true, c, fr);
            double best = 1e30;
            for (int r = 0; r < 3; ++r) { const double t = now(); eng.encode(x.data(), (int64_t) x.size(), true, c, fr); best = std::min(best, now() - t); }
            const double sec = (double) x.size() / m.sample_rate;
            printf("encode padded   %.2f s audio -> %lld frames: %.3f s (RTF %.3f, %.1fx realtime)\n", sec, (long long) fr,
                   best, best / sec, sec / best);
        }
        return 0;
    }
    return usage();
}
