// Portable "tap-GEMM" backend. With channels-last activations a K-tap 1-D convolution is a sum of
// K matrix products on row-strided views, so no im2col buffer is needed:
//   conv : y[t]          += x[t*s + k*d - pad] . W_k      (A rows spaced s*IC apart)
//   convT: y[t*s + k - pad] += x[t] . W_k                 (C rows spaced s*OC apart)
// with W_k = W[:, :, k] packed once as [IC, OC]. Each tap is one row-major SGEMM with beta = 1.
#include "backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(DAC_SGEMM_ACCELERATE)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#elif defined(DAC_SGEMM_CBLAS)
#include <cblas.h>
#elif defined(DAC_SGEMM_DNNL)
#include <dnnl.h>
#endif

namespace dacn {
namespace {

// C[M,N] += A[M,K] * B[K,N], row-major with leading dimensions.
void sgemm_acc(int64_t M, int64_t N, int64_t K, const float * A, int64_t lda, const float * B, int64_t ldb, float * C,
               int64_t ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;
#if defined(DAC_SGEMM_ACCELERATE) || defined(DAC_SGEMM_CBLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) M, (int) N, (int) K, 1.0f, A, (int) lda, B, (int) ldb,
                1.0f, C, (int) ldc);
#elif defined(DAC_SGEMM_DNNL)
    dnnl_sgemm('N', 'N', M, N, K, 1.0f, A, lda, B, ldb, 1.0f, C, ldc);
#else
    // Reference fallback: cache-blocked, parallel over rows. Correct everywhere, not fast.
    parallel_for(M, [&](int64_t m0, int64_t m1) {
        for (int64_t m = m0; m < m1; ++m) {
            float * c = C + m * ldc;
            const float * a = A + m * lda;
            for (int64_t k = 0; k < K; ++k) {
                const float av = a[k];
                const float * b = B + k * ldb;
                for (int64_t n = 0; n < N; ++n) c[n] += av * b[n];
            }
        }
    });
#endif
}

class gemm_backend final : public backend {
public:
    const char * name() const override {
#if defined(DAC_SGEMM_ACCELERATE)
        return "tap-GEMM (Accelerate)";
#elif defined(DAC_SGEMM_CBLAS)
        return "tap-GEMM (CBLAS)";
#elif defined(DAC_SGEMM_DNNL)
        return "tap-GEMM (dnnl_sgemm)";
#else
        return "tap-GEMM (reference)";
#endif
    }

    std::function<void()> conv(const conv_params & p, view src, view dst) override {
        if (src.C != p.IC || dst.C != p.OC) throw error("conv channel mismatch");
        if (conv_out_len(p, src.T) != dst.T) throw error("conv length mismatch");
        // pack W_k as [IC, OC] for every tap
        auto packed = std::make_shared<fbuf>(alloc_floats((size_t) (p.K * p.IC * p.OC)));
        float * wk = packed->get();
        for (int64_t k = 0; k < p.K; ++k)
            for (int64_t ic = 0; ic < p.IC; ++ic)
                for (int64_t oc = 0; oc < p.OC; ++oc)
                    wk[(k * p.IC + ic) * p.OC + oc] = p.transposed ? p.w[(ic * p.OC + oc) * p.K + k]   // [IC][OC][K]
                                                                   : p.w[(oc * p.IC + ic) * p.K + k];  // [OC][IC][K]
        const conv_params q = p;
        return [q, src, dst, packed]() { run(q, src, dst, packed->get()); };
    }

private:
    static void run(const conv_params & p, const view & x, const view & y, const float * wk) {
        const int64_t To = y.T, Ti = x.T, IC = p.IC, OC = p.OC, s = p.stride;
        // init: y = bias (or y += bias for residual sum)
        parallel_for(To, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                float * row = y.ptr + t * OC;
                if (p.sum_into) for (int64_t c = 0; c < OC; ++c) row[c] += p.b[c];
                else memcpy(row, p.b, (size_t) OC * sizeof(float));
            }
        });
        for (int64_t k = 0; k < p.K; ++k) {
            const float * W = wk + k * IC * OC;
            if (!p.transposed) {
                const int64_t off = k * p.dil - p.pad;              // input row = t*s + off
                const int64_t t0 = off >= 0 ? 0 : (-off + s - 1) / s;
                const int64_t lim = Ti - off;                         // need t*s < lim
                const int64_t t1 = lim <= 0 ? 0 : std::min(To, (lim + s - 1) / s);
                if (t1 > t0) sgemm_acc(t1 - t0, OC, IC, x.ptr + (t0 * s + off) * IC, s * IC, W, OC, y.ptr + t0 * OC, OC);
            } else {
                const int64_t off = k * p.dil - p.pad;              // output row = t*s + off
                const int64_t t0 = off >= 0 ? 0 : (-off + s - 1) / s;
                const int64_t lim = To - off;
                const int64_t t1 = lim <= 0 ? 0 : std::min(Ti, (lim + s - 1) / s);
                if (t1 > t0) sgemm_acc(t1 - t0, OC, IC, x.ptr + t0 * IC, IC, W, OC, y.ptr + (t0 * s + off) * OC, s * OC);
            }
        }
        if (p.tanh)
            parallel_for(To * OC, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; ++i) y.ptr[i] = std::tanh(y.ptr[i]); });
    }
};

}  // namespace

std::unique_ptr<backend> make_backend() { return std::make_unique<gemm_backend>(); }

}  // namespace dacn
