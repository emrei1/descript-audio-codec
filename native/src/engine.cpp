#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dacn {

struct engine::plan {
    std::vector<fbuf> bufs;
    std::vector<std::function<void()>> steps;
    view in, out;
    // encoder: RVQ output; decoder: code rows for the quantizer lookup
    std::vector<int32_t> codes;
    const int32_t * const * code_rows = nullptr;
    int n_cb = 0;

    view alloc(int64_t C, int64_t T) {
        bufs.push_back(alloc_floats((size_t) (C * T)));
        return {bufs.back().get(), C, T};
    }
};

engine::engine(const model & m) : m_(m), be_(make_backend()) {}
engine::~engine() = default;

const char * engine::backend_name() const { return be_->name(); }

namespace {

// Graph-building helper shared by the encoder and decoder.
struct builder {
    const model & m;
    backend & be;
    void * pl;   // engine::plan, kept opaque here
    bool padded;
    std::function<view(int64_t, int64_t)> alloc;
    std::vector<std::function<void()>> & steps;

    conv_params params(const std::string & name, int64_t stride, int64_t dil, bool transposed) const {
        const tensor & w = m.get(name + ".weight");
        conv_params p;
        p.K = w.ne[0];
        p.IC = transposed ? w.ne[2] : w.ne[1];
        p.OC = transposed ? w.ne[1] : w.ne[2];
        p.stride = stride;
        p.dil = dil;
        p.transposed = transposed;
        p.w = w.data;
        p.b = m.get(name + ".bias").data;
        // PyTorch padding: (K-1)*d/2 for "same" convs, ceil(stride/2) for the (de)sampling convs
        p.pad = !padded ? 0 : (stride > 1 ? (stride + 1) / 2 : (p.K - 1) * dil / 2);
        return p;
    }

    view conv(const std::string & name, const view & src, int64_t stride = 1, int64_t dil = 1, bool transposed = false,
              const view * sum_into = nullptr, bool tanh = false) {
        conv_params p = params(name, stride, dil, transposed);
        if (src.C != p.IC) throw error("channel mismatch at " + name);
        const int64_t To = conv_out_len(p, src.T);
        if (To <= 0) throw error("input too short for " + name);
        p.sum_into = sum_into != nullptr;
        p.tanh = tanh;
        view dst = sum_into ? *sum_into : alloc(p.OC, To);
        if (dst.T != To || dst.C != p.OC) throw error("residual shape mismatch at " + name);
        steps.push_back(be.conv(p, src, dst));
        return dst;
    }

    view snake_op(const std::string & name, const view & src, const view * dst_opt = nullptr) {
        const float * a = m.get(name + ".alpha").data;
        const float * ia = m.get(name + ".inv_alpha").data;
        const view dst = dst_opt ? *dst_opt : alloc(src.C, src.T);
        const view s = src;
        steps.push_back([s, dst, a, ia]() { snake(s.ptr, dst.ptr, a, ia, s.T, s.C); });
        return dst;
    }

    // ResidualUnit: x + conv2(snake2(conv1(snake1(x)))), x center-cropped when unpadded.
    view res_unit(const std::string & name, view x, int64_t dil) {
        view a = snake_op(name + ".snake1", x);
        view h = conv(name + ".conv1", a, 1, dil);
        snake_op(name + ".snake2", h, &h);
        const int64_t crop = (x.T - h.T) / 2;
        view xr{x.ptr + crop * x.C, x.C, h.T};
        return conv(name + ".conv2", h, 1, 1, false, &xr);
    }
};

int64_t res_len(int64_t T, bool padded) { return padded ? T : T - 6 * (1 + 3 + 9); }

}  // namespace

int64_t engine::encoder_out_len(int64_t N, bool padded) const {
    auto c = [&](int64_t T, int64_t K, int64_t s, int64_t pad) { return (T + 2 * pad - (K - 1) - 1) / s + 1; };
    int64_t T = c(N, 7, 1, padded ? 3 : 0);
    for (int s : m_.encoder_rates) T = c(res_len(T, padded), 2 * s, s, padded ? (s + 1) / 2 : 0);
    return c(T, 3, 1, padded ? 1 : 0);
}

int64_t engine::decoder_out_len(int64_t L, bool padded) const {
    int64_t T = L + (padded ? 0 : -6);
    for (int s : m_.decoder_rates) T = res_len((T - 1) * s - 2 * (padded ? (s + 1) / 2 : 0) + 2 * s, padded);
    return T + (padded ? 0 : -6);
}

engine::plan & engine::enc_plan(int64_t N, bool padded) {
    auto & slot = enc_[{N, padded}];
    if (slot) return *slot;
    if (!m_.has_encoder()) throw error("this model file has no encoder (convert it with scripts/convert_weights.py)");
    auto pp = std::make_unique<plan>();
    plan & p = *pp;
    builder b{m_, *be_, &p, padded, [&p](int64_t C, int64_t T) { return p.alloc(C, T); }, p.steps};

    p.in = p.alloc(1, N);
    view x = b.conv("enc.conv_in", p.in, 1, 1);
    for (size_t i = 0; i < m_.encoder_rates.size(); ++i) {
        const std::string pre = "enc.b" + std::to_string(i);
        const int dils[3] = {1, 3, 9};
        for (int r = 0; r < 3; ++r) x = b.res_unit(pre + ".res" + std::to_string(r), x, dils[r]);
        view sn = b.snake_op(pre + ".snake", x);
        x = b.conv(pre + ".conv", sn, m_.encoder_rates[i], 1);
    }
    view sn = b.snake_op("enc.snake_out", x);
    p.out = b.conv("enc.conv_out", sn, 1, 1);   // latent [T, latent_dim]
    if (p.out.T != encoder_out_len(N, padded)) throw error("internal encoder length mismatch");

    // Residual vector quantization (quantizer.forward in eval mode), per frame:
    //   e = normalize(W_in r + b_in); idx = argmax_j e . normalize(codebook_j)
    //   r -= out_proj(codebook[idx]) = table[idx] + b_out
    const int n_cb = m_.n_codebooks, D = m_.codebook_dim, S = m_.codebook_size, Lt = m_.latent_dim;
    std::vector<const float *> in_w, in_b, out_b, table;
    auto ncb = std::make_shared<std::vector<float>>((size_t) n_cb * S * D);   // L2-normalized codebooks
    auto cn2 = std::make_shared<std::vector<float>>((size_t) n_cb * S);       // their squared norms
    for (int k = 0; k < n_cb; ++k) {
        const std::string q = "quant." + std::to_string(k);
        in_w.push_back(m_.get(q + ".in_w").data);
        in_b.push_back(m_.get(q + ".in_b").data);
        out_b.push_back(m_.get(q + ".out_b").data);
        table.push_back(m_.get(q + ".table").data);
        const float * cb = m_.get(q + ".codebook").data;
        for (int j = 0; j < S; ++j) {
            double n2 = 0;
            for (int d = 0; d < D; ++d) n2 += (double) cb[j * D + d] * cb[j * D + d];
            const float inv = 1.0f / std::max((float) std::sqrt(n2), 1e-12f);
            float q2 = 0;
            for (int d = 0; d < D; ++d) {
                const float v = cb[j * D + d] * inv;
                (*ncb)[((size_t) k * S + j) * D + d] = v;
                q2 += v * v;
            }
            (*cn2)[(size_t) k * S + j] = q2;
        }
    }
    const view z = p.out;
    plan * P = &p;
    p.steps.push_back([=]() {
        const int64_t T = z.T;
        P->codes.assign((size_t) n_cb * T, 0);
        parallel_for(T, [&](int64_t t0, int64_t t1) {
            std::vector<float> r((size_t) Lt), e((size_t) D);
            for (int64_t t = t0; t < t1; ++t) {
                memcpy(r.data(), z.ptr + t * Lt, (size_t) Lt * sizeof(float));
                for (int k = 0; k < n_cb; ++k) {
                    double nn = 0;
                    for (int d = 0; d < D; ++d) {
                        const float * w = in_w[k] + (int64_t) d * Lt;
                        float acc = 0;
                        for (int c = 0; c < Lt; ++c) acc += w[c] * r[c];
                        e[d] = acc + in_b[k][d];
                        nn += (double) e[d] * e[d];
                    }
                    const float inv = 1.0f / std::max((float) std::sqrt(nn), 1e-12f);
                    // reference: argmax_j -(|e|^2 - 2 e.c_j + |c_j|^2); |e|^2 is constant per frame
                    for (int d = 0; d < D; ++d) e[d] *= inv;
                    int best = 0;
                    float best_score = -INFINITY;
                    const float * cbk = ncb->data() + (size_t) k * S * D;
                    const float * c2 = cn2->data() + (size_t) k * S;
                    for (int j = 0; j < S; ++j) {
                        float dot = 0;
                        for (int d = 0; d < D; ++d) dot += e[d] * cbk[j * D + d];
                        const float score = 2.0f * dot - c2[j];
                        if (score > best_score) { best_score = score; best = j; }
                    }
                    P->codes[(size_t) k * T + t] = best;
                    const float * row = table[k] + (int64_t) best * Lt;
                    for (int c = 0; c < Lt; ++c) r[c] -= row[c] + out_b[k][c];
                }
            }
        });
    });
    slot = std::move(pp);
    return *slot;
}

engine::plan & engine::dec_plan(int64_t L, bool padded) {
    auto & slot = dec_[{L, padded}];
    if (slot) return *slot;
    auto pp = std::make_unique<plan>();
    plan & p = *pp;
    builder b{m_, *be_, &p, padded, [&p](int64_t C, int64_t T) { return p.alloc(C, T); }, p.steps};

    // quantizer.from_codes with folded tables: z[t] = sum_k table_k[code_k[t]] + sum of out_proj biases
    const int Lt = m_.latent_dim, S = m_.codebook_size;
    p.in = p.alloc(Lt, L);
    {
        std::vector<const float *> table;
        for (int k = 0; k < m_.n_codebooks; ++k) table.push_back(m_.get("quant." + std::to_string(k) + ".table").data);
        const float * bias = m_.get("quant.bias").data;
        const view z = p.in;
        plan * P = &p;
        p.steps.push_back([=]() {
            parallel_for(z.T, [&](int64_t t0, int64_t t1) {
                for (int64_t t = t0; t < t1; ++t) {
                    float * zt = z.ptr + t * Lt;
                    memcpy(zt, bias, (size_t) Lt * sizeof(float));
                    for (int k = 0; k < P->n_cb; ++k) {
                        const int idx = std::min(std::max(P->code_rows[k][t], 0), S - 1);
                        const float * row = table[k] + (int64_t) idx * Lt;
                        for (int c = 0; c < Lt; ++c) zt[c] += row[c];
                    }
                }
            });
        });
    }
    view x = b.conv("dec.conv_in", p.in, 1, 1);
    for (size_t i = 0; i < m_.decoder_rates.size(); ++i) {
        const std::string pre = "dec.b" + std::to_string(i);
        view sn = b.snake_op(pre + ".snake", x);
        x = b.conv(pre + ".convt", sn, m_.decoder_rates[i], 1, /*transposed=*/true);
        const int dils[3] = {1, 3, 9};
        for (int r = 0; r < 3; ++r) x = b.res_unit(pre + ".res" + std::to_string(r), x, dils[r]);
    }
    view sn = b.snake_op("dec.snake_out", x);
    p.out = b.conv("dec.conv_out", sn, 1, 1, false, nullptr, /*tanh=*/true);
    if (p.out.T != decoder_out_len(L, padded)) throw error("internal decoder length mismatch");
    slot = std::move(pp);
    return *slot;
}

void engine::encode(const float * audio, int64_t N, bool padded, std::vector<int32_t> & codes, int64_t & frames) {
    plan & p = enc_plan(N, padded);
    memcpy(p.in.ptr, audio, (size_t) N * sizeof(float));
    for (auto & s : p.steps) s();
    be_->sync();
    frames = p.out.T;
    codes = p.codes;
}

void engine::decode(const int32_t * const * codes, int n_cb, int64_t L, bool padded, std::vector<float> & out) {
    if (n_cb <= 0 || n_cb > m_.n_codebooks) throw error("codes have more codebooks than the model");
    plan & p = dec_plan(L, padded);
    p.code_rows = codes;
    p.n_cb = n_cb;
    for (auto & s : p.steps) s();
    be_->sync();
    out.assign(p.out.ptr, p.out.ptr + p.out.T);
}

}  // namespace dacn
