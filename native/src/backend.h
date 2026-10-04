// Convolution backends. Activations are channels-last [T, C] (x[t*C + c]) everywhere.
//   oneDNN  (x86-64): JIT brgemm convolution / deconvolution primitives, fused bias, residual
//                     sum and tanh post-ops; weights reordered once into the preferred layout.
//   gemm    (other) : "tap-GEMM" - a K-tap conv is K SGEMM calls on strided row views, no im2col.
//                     SGEMM from Apple Accelerate, a CBLAS (OpenBLAS), or oneDNN's dnnl_sgemm.
#pragma once

#include "common.h"

namespace dacn {

struct view {
    float * ptr = nullptr;
    int64_t C = 0, T = 0;
};

struct conv_params {
    int64_t K = 1, IC = 0, OC = 0, stride = 1, dil = 1, pad = 0;
    bool transposed = false;   // ConvTranspose1d; weights in PyTorch [IC, OC, K] layout (else [OC, IC, K])
    bool sum_into = false;     // dst += conv(src) + bias (residual connection), else dst = conv + bias
    bool tanh = false;         // apply tanh to the result
    const float * w = nullptr;
    const float * b = nullptr;
};

// Output length of a conv / transposed conv with symmetric padding.
inline int64_t conv_out_len(const conv_params & p, int64_t T) {
    if (p.transposed) return (T - 1) * p.stride - 2 * p.pad + p.dil * (p.K - 1) + 1;
    return (T + 2 * p.pad - p.dil * (p.K - 1) - 1) / p.stride + 1;
}

class backend {
public:
    virtual ~backend() = default;
    // Builds an executable step; src/dst buffers must stay valid. Throws dacn::error.
    virtual std::function<void()> conv(const conv_params & p, view src, view dst) = 0;
    virtual void sync() {}
    virtual const char * name() const = 0;
};

std::unique_ptr<backend> make_backend();

// snake(x) = x + inv_alpha[c] * sin(alpha[c] * x)^2 over [T, C]; dst may alias src.
void snake(const float * src, float * dst, const float * alpha, const float * inv_alpha, int64_t T, int64_t C);

}  // namespace dacn
