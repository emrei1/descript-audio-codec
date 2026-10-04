#include "codec.h"

#include "dsp.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>

namespace dacn {
namespace {

struct layer { bool transposed; int64_t k, s, d; };

// Every Conv1d / ConvTranspose1d of the PyTorch model in module order (encoder, quantizer, decoder).
std::vector<layer> conv_layers(const model & m) {
    std::vector<layer> v;
    auto res = [&] { for (int d : {1, 3, 9}) { v.push_back({false, 7, 1, d}); v.push_back({false, 1, 1, 1}); } };
    v.push_back({false, 7, 1, 1});
    for (int s : m.encoder_rates) { res(); v.push_back({false, 2 * s, s, 1}); }
    v.push_back({false, 3, 1, 1});
    for (int i = 0; i < m.n_codebooks; ++i) { v.push_back({false, 1, 1, 1}); v.push_back({false, 1, 1, 1}); }
    v.push_back({false, 7, 1, 1});
    for (int s : m.decoder_rates) { v.push_back({true, 2 * s, s, 1}); res(); }
    v.push_back({false, 7, 1, 1});
    return v;
}

double db_gain(double db) { return std::exp(db * (std::log(10.0) / 20.0)); }

}  // namespace

int64_t codec_output_length(const model & m, int64_t input_length) {
    double L = (double) input_length;
    for (const layer & l : conv_layers(m)) {
        if (!l.transposed) L = (L - l.d * (l.k - 1) - 1) / l.s + 1;
        else L = (L - 1) * l.s + l.d * (l.k - 1) + 1;
        L = std::floor(L);
    }
    return (int64_t) L;
}

int64_t codec_delay(const model & m) {
    const int64_t l_out = codec_output_length(m, 0);
    double L = (double) l_out;
    auto layers = conv_layers(m);
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        if (it->transposed) L = (L - it->d * (it->k - 1) - 1) / it->s + 1;
        else L = (L - 1) * it->s + it->d * (it->k - 1) + 1;
        L = std::ceil(L);
    }
    const int64_t l_in = (int64_t) L;
    const int64_t diff = l_in - l_out;
    return diff >= 0 ? diff / 2 : -((-diff + 1) / 2);   // Python floor division
}

bool compress(engine & eng, const audio_buffer & in, dac_file & out, std::string & err, const compress_options & opt,
              progress * prog) {
    try {
        const model & m = eng.weights();
        const int sr = m.sample_rate, hop_len = m.hop_length, nac = in.channels;
        out = dac_file{};
        out.original_length = in.frames;
        out.sample_rate = in.sample_rate;
        out.channels = nac;

        // resample every channel to the model rate
        std::vector<std::vector<float>> ch((size_t) nac);
        std::vector<float> tmp((size_t) in.frames);
        for (int c = 0; c < nac; ++c) {
            for (int64_t t = 0; t < in.frames; ++t) tmp[(size_t) t] = in.samples[(size_t) (t * nac + c)];
            ch[(size_t) c] = resample_frac(tmp.data(), in.frames, in.sample_rate, sr);
        }
        const int64_t nt = (int64_t) ch[0].size();

        // input_db = joint loudness; normalize to normalize_db; ensure_max_of_audio per channel
        std::vector<const float *> ptrs;
        for (auto & c : ch) ptrs.push_back(c.data());
        out.input_db = loudness_lufs(ptrs, nt, sr);
        const float g = (float) db_gain(opt.normalize_db - out.input_db);
        for (auto & c : ch) {
            float peak = 0;
            for (float & v : c) { v *= g; peak = std::max(peak, std::fabs(v)); }
            if (peak > 1.0f) for (float & v : c) v /= peak;
        }

        int64_t n_samples, hop, delay = 0;
        if ((double) nt / sr <= opt.win_duration) {   // unchunked
            out.padding = true;
            n_samples = nt;
            hop = nt;
        } else {                                      // chunked inference, unpadded convs
            out.padding = false;
            delay = codec_delay(m);
            n_samples = (int64_t) std::ceil((double) (int64_t) (opt.win_duration * sr) / hop_len) * hop_len;
            hop = codec_output_length(m, n_samples);
            if (hop <= 0) throw error("window too short");
        }
        const int64_t n_chunks = (nt + hop - 1) / hop;
        if (prog) { prog->total = n_chunks * nac; prog->done = 0; }
        const int n_cb = opt.n_quantizers > 0 ? std::min(opt.n_quantizers, m.n_codebooks) : m.n_codebooks;

        // per channel: zero_pad(delay, delay), then windows [i, i + n_samples) for i in range(0, nt, hop)
        std::vector<std::vector<std::vector<int32_t>>> per_chan((size_t) nac);   // [c][k] -> codes
        std::vector<float> win;
        std::vector<int32_t> codes;
        int64_t chunk_len = 0;
        for (int c = 0; c < nac; ++c) {
            per_chan[(size_t) c].assign((size_t) n_cb, {});
            for (int64_t i = 0; i < nt; i += hop) {
                if (prog && prog->cancel) throw error("cancelled");
                // preprocess(): right-pad to a multiple of hop_length
                const int64_t len = (n_samples + hop_len - 1) / hop_len * hop_len;
                win.assign((size_t) len, 0.0f);
                for (int64_t j = 0; j < n_samples; ++j) {
                    const int64_t src = i + j - delay;   // index into the unpadded signal
                    if (src >= 0 && src < nt) win[(size_t) j] = ch[(size_t) c][(size_t) src];
                }
                int64_t frames = 0;
                eng.encode(win.data(), len, out.padding, codes, frames);
                for (int k = 0; k < n_cb; ++k)
                    per_chan[(size_t) c][(size_t) k].insert(per_chan[(size_t) c][(size_t) k].end(),
                                                            codes.begin() + k * frames, codes.begin() + (k + 1) * frames);
                chunk_len = frames;
                if (prog) ++prog->done;
            }
        }
        out.B = nac;
        out.n_codebooks = n_cb;
        out.T = (int64_t) per_chan[0][0].size();
        out.chunk_length = chunk_len;
        out.codes.reserve((size_t) (out.B * n_cb * out.T));
        for (int c = 0; c < nac; ++c)
            for (int k = 0; k < n_cb; ++k)
                out.codes.insert(out.codes.end(), per_chan[(size_t) c][(size_t) k].begin(), per_chan[(size_t) c][(size_t) k].end());
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

bool decompress(engine & eng, const dac_file & f, decompress_result & out, std::string & err,
                const decompress_options & opt, progress * prog) {
    try {
        using clk = std::chrono::steady_clock;
        const model & m = eng.weights();
        if (f.n_codebooks > m.n_codebooks) throw error("file uses more codebooks than this model (different DAC variant?)");
        for (int32_t c : f.codes)
            if (c >= m.codebook_size) throw error("code index out of range for this model");
        if (f.B != f.channels) throw error("multi-item .dac batches are not supported");

        const int64_t chunk = f.chunk_length;
        if (prog) { prog->total = f.B * ((f.T + chunk - 1) / chunk); prog->done = 0; }
        const int sr = m.sample_rate;
        std::vector<std::vector<float>> chans((size_t) f.B);
        std::vector<const int32_t *> rows((size_t) f.n_codebooks);
        std::vector<float> piece;
        double t_net = 0;

        // codes[..., i:i+chunk_length] -> from_codes -> decode, concatenated
        for (int64_t b = 0; b < f.B; ++b) {
            for (int64_t i = 0; i < f.T; i += chunk) {
                if (prog && prog->cancel) throw error("cancelled");
                const int64_t L = std::min(chunk, f.T - i);
                for (int64_t k = 0; k < f.n_codebooks; ++k) rows[(size_t) k] = f.code_row(b, k) + i;
                const auto t0 = clk::now();
                eng.decode(rows.data(), (int) f.n_codebooks, L, f.padding, piece);
                t_net += std::chrono::duration<double>(clk::now() - t0).count();
                chans[(size_t) b].insert(chans[(size_t) b].end(), piece.begin(), piece.end());
                if (prog) ++prog->done;
            }
        }

        // normalize to input_db
        if (opt.reference) {
            for (auto & rec : chans) {
                const float g = (float) db_gain(f.input_db - loudness_lufs(rec.data(), (int64_t) rec.size(), sr));
                for (float & v : rec) v *= g;
            }
        } else {
            std::vector<const float *> ptrs;
            int64_t len = INT64_MAX;
            for (auto & rec : chans) { ptrs.push_back(rec.data()); len = std::min<int64_t>(len, (int64_t) rec.size()); }
            const float g = (float) db_gain(f.input_db - loudness_lufs(ptrs, len, sr));
            for (auto & rec : chans) for (float & v : rec) v *= g;
        }

        // resample to the original rate, crop to original_length, interleave
        audio_buffer & a = out.audio;
        a.channels = (int) f.channels;
        a.sample_rate = (int) f.sample_rate;
        a.frames = f.original_length;
        a.samples.assign((size_t) (a.frames * a.channels), 0.0f);
        for (int64_t c = 0; c < f.channels; ++c) {
            const auto & rec = chans[(size_t) c];
            const std::vector<float> rs = resample_frac(rec.data(), (int64_t) rec.size(), sr, (int) f.sample_rate);
            const int64_t n = std::min<int64_t>(a.frames, (int64_t) rs.size());
            for (int64_t t = 0; t < n; ++t) a.samples[(size_t) (t * a.channels + c)] = rs[(size_t) t];
        }
        out.limiter_db = opt.reference ? 0.0 : limit_peaks(a.samples.data(), a.frames, a.channels, a.sample_rate, opt.ceiling);
        out.neural_seconds = t_net;
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

}  // namespace dacn
