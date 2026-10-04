// Signal processing used around the codec, ported from descript-audiotools / julius so results
// match the Python reference: BS.1770 loudness, sinc resampling, plus a peak limiter.
#include "dsp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace dacn {
namespace {

constexpr double PI = 3.14159265358979323846;

// pyloudnorm IIRfilter biquads (K-weighting = high shelf then high pass), RBJ-cookbook style,
// exactly as pyloudnorm.IIRfilter.generate_coefficients computes them.
struct biquad { double b[3], a[3]; };

biquad k_filter(int rate, bool shelf) {
    const double G = shelf ? 4.0 : 0.0;
    const double Q = shelf ? 1.0 / std::sqrt(2.0) : 0.5;
    const double fc = shelf ? 1500.0 : 38.0;
    const double A = std::pow(10.0, G / 40.0);
    const double w0 = 2.0 * PI * (fc / rate);
    const double alpha = std::sin(w0) / (2.0 * Q);
    double b0, b1, b2, a0, a1, a2;
    if (shelf) {   // high_shelf
        b0 = A * ((A + 1) + (A - 1) * std::cos(w0) + 2 * std::sqrt(A) * alpha);
        b1 = -2 * A * ((A - 1) + (A + 1) * std::cos(w0));
        b2 = A * ((A + 1) + (A - 1) * std::cos(w0) - 2 * std::sqrt(A) * alpha);
        a0 = (A + 1) - (A - 1) * std::cos(w0) + 2 * std::sqrt(A) * alpha;
        a1 = 2 * ((A - 1) - (A + 1) * std::cos(w0));
        a2 = (A + 1) - (A - 1) * std::cos(w0) - 2 * std::sqrt(A) * alpha;
    } else {       // high_pass
        b0 = (1 + std::cos(w0)) / 2;
        b1 = -(1 + std::cos(w0));
        b2 = (1 + std::cos(w0)) / 2;
        a0 = 1 + alpha;
        a1 = -2 * std::cos(w0);
        a2 = 1 - alpha;
    }
    return {{b0 / a0, b1 / a0, b2 / a0}, {1.0, a1 / a0, a2 / a0}};
}

void lfilter(const biquad & q, std::vector<double> & x) {
    double z1 = 0, z2 = 0;   // transposed direct form II
    for (double & v : x) {
        const double y = q.b[0] * v + z1;
        z1 = q.b[1] * v - q.a[1] * y + z2;
        z2 = q.b[2] * v - q.a[2] * y;
        v = y;
    }
}

}  // namespace

double loudness_lufs(const float * x, int64_t n, int rate) {
    return loudness_lufs(std::vector<const float *>{x}, n, rate);
}

double loudness_lufs(const std::vector<const float *> & chans, int64_t n, int rate) {
    // AudioSignal.loudness(): zero-pad to at least 0.5 s for the measurement
    int64_t len = n;
    const double dur = (double) n / rate;
    if (dur < 0.5) len = n + (int64_t) ((0.5 - dur) * rate);

    // gating blocks: 400 ms, 75% overlap, julius.core.unfold zero-pads the tail
    const double Tg = 0.4;
    const int64_t K = (int64_t) (Tg * rate);
    const int64_t S = (int64_t) (Tg * rate * 0.25);
    const int64_t nf = (int64_t) std::ceil((double) (std::max(len, K) - K) / S) + 1;
    std::vector<double> z((size_t) nf, 0.0), l((size_t) nf);

    // per channel: K-weighting, then G-weighted block mean squares summed across channels
    static const double G[5] = {1.0, 1.0, 1.0, 1.41, 1.41};
    std::vector<double> d((size_t) len);
    for (size_t c = 0; c < chans.size(); ++c) {
        std::fill(d.begin(), d.end(), 0.0);
        for (int64_t i = 0; i < n; ++i) d[(size_t) i] = chans[c][i];
        lfilter(k_filter(rate, true), d);
        lfilter(k_filter(rate, false), d);
        const double g = c < 5 ? G[c] : 1.0;
        for (int64_t f = 0; f < nf; ++f) {
            double acc = 0;
            for (int64_t k = f * S; k < f * S + K && k < len; ++k) acc += d[(size_t) k] * d[(size_t) k];
            z[(size_t) f] += g * acc / (Tg * rate);
        }
    }
    for (int64_t f = 0; f < nf; ++f) l[(size_t) f] = -0.691 + 10.0 * std::log10(z[(size_t) f]);
    const double Ga = -70.0;
    double sum = 0; int64_t cnt = 0;
    for (int64_t f = 0; f < nf; ++f) if (l[(size_t) f] > Ga) { sum += z[(size_t) f]; ++cnt; }
    const double zavg = sum / (double) cnt;   // 0/0 -> NaN, as in the reference
    const double Gr = -0.691 + 10.0 * std::log10(zavg) - 10.0;
    sum = 0; cnt = 0;
    for (int64_t f = 0; f < nf; ++f)
        if (l[(size_t) f] > Ga && l[(size_t) f] > Gr) { sum += z[(size_t) f]; ++cnt; }
    double zg = sum / (double) cnt;
    if (std::isnan(zg)) zg = 0;
    const double lufs = -0.691 + 10.0 * std::log10(zg);
    return std::isnan(lufs) ? -70.0 : std::max(lufs, -70.0);
}

double limit_peaks(float * x, int64_t frames, int channels, int sample_rate, float ceiling, double lookahead_ms,
                   double release_ms) {
    const int64_t W = std::max<int64_t>(1, (int64_t) (lookahead_ms * 1e-3 * sample_rate));
    // required gain per frame
    std::vector<float> req((size_t) frames);
    bool any = false;
    for (int64_t t = 0; t < frames; ++t) {
        float pk = 0;
        for (int c = 0; c < channels; ++c) pk = std::max(pk, std::fabs(x[t * channels + c]));
        req[(size_t) t] = pk > ceiling ? ceiling / pk : 1.0f;
        any |= pk > ceiling;
    }
    if (!any) return 0.0;
    // m[t] = min(req[t .. t+W-1]) (sliding-window minimum, monotonic deque)
    std::vector<float> m((size_t) frames);
    std::vector<int64_t> dq((size_t) frames + 1);
    size_t head = 0, tail = 0;
    for (int64_t t = frames - 1; t >= 0; --t) {
        while (tail > head && req[(size_t) dq[tail - 1]] >= req[(size_t) t]) --tail;
        dq[tail++] = t;
        while (dq[head] > t + W - 1) ++head;
        m[(size_t) t] = req[(size_t) dq[head]];
    }
    // release: gain recovers exponentially toward 1, but never above m
    const double rel = 1.0 - std::exp(-1.0 / (release_ms * 1e-3 * sample_rate));
    std::vector<float> e((size_t) frames);
    double g = 1.0;
    for (int64_t t = 0; t < frames; ++t) {
        g = std::min<double>(m[(size_t) t], g + (1.0 - g) * rel);
        e[(size_t) t] = (float) g;
    }
    // attack: W-sample moving average of e (<= m at every peak, since e <= v on [p-W+1, p])
    double acc = (double) W, worst = 1.0;   // history starts at unity gain
    std::vector<float> hist((size_t) W, 1.0f);
    for (int64_t t = 0; t < frames; ++t) {
        acc += e[(size_t) t] - hist[(size_t) (t % W)];
        hist[(size_t) (t % W)] = e[(size_t) t];
        const float s = (float) std::min(1.0, acc / (double) W);
        worst = std::min<double>(worst, s);
        for (int c = 0; c < channels; ++c) {
            float & v = x[t * channels + c];
            v *= s;
            v = std::max(-ceiling, std::min(ceiling, v));   // numerical safety
        }
    }
    return -20.0 * std::log10(worst);
}

std::vector<float> resample_frac(const float * x, int64_t n, int old_sr, int new_sr) {
    if (old_sr == new_sr) return std::vector<float>(x, x + n);
    auto gcd = [](int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; };
    const int g = gcd(old_sr, new_sr);
    const int o = old_sr / g, nw = new_sr / g;
    const int zeros = 24;
    const double rolloff = 0.945;
    const double sr = std::min(o, nw) * rolloff;
    const int width = (int) std::ceil(zeros * o / sr);
    const int K = 2 * width + o;

    // kernels[i][j], i = output phase, j over idx = -width .. width+o-1
    std::vector<float> kern((size_t) nw * K);
    for (int i = 0; i < nw; ++i) {
        double ksum = 0;
        std::vector<double> kk(K);
        for (int j = 0; j < K; ++j) {
            const double idx = -width + j;
            double t = (-(double) i / nw + idx / o) * sr;
            t = std::max(-(double) zeros, std::min((double) zeros, t)) * PI;
            const double win = std::cos(t / zeros / 2);
            const double sinc = t == 0 ? 1.0 : std::sin(t) / t;
            kk[j] = sinc * win * win;
            ksum += kk[j];
        }
        for (int j = 0; j < K; ++j) kern[(size_t) i * K + j] = (float) (kk[j] / ksum);
    }

    // replicate-pad left by width, right by width + o; conv1d with stride o
    const int64_t padded = n + 2 * width + o;
    auto xp = [&](int64_t k) -> float {
        int64_t s = k - width;
        s = s < 0 ? 0 : (s >= n ? n - 1 : s);
        return x[s];
    };
    const int64_t frames = (padded - K) / o + 1;
    const int64_t out_len = (int64_t) std::floor((double) nw * n / o);
    std::vector<float> y((size_t) out_len);
    parallel_for(frames, [&](int64_t f0, int64_t f1) {
        for (int64_t f = f0; f < f1; ++f) {
            for (int i = 0; i < nw; ++i) {
                const int64_t oi = f * nw + i;
                if (oi >= out_len) break;
                const float * kr = &kern[(size_t) i * K];
                const int64_t base = f * o;
                double acc = 0;
                if (base - width >= 0 && base - width + K <= n) {
                    const float * xs = x + base - width;
                    for (int j = 0; j < K; ++j) acc += (double) kr[j] * xs[j];
                } else {
                    for (int j = 0; j < K; ++j) acc += (double) kr[j] * xp(base + j);
                }
                y[(size_t) oi] = (float) acc;
            }
        }
    });
    return y;
}

}  // namespace dacn
