// AVX2 + FMA Snake kernel (this file alone is compiled with -mavx2 -mfma / /arch:AVX2).
#include "backend.h"

#include <immintrin.h>

namespace dacn {

static inline __m256 sin2_ps(__m256 x) {
    const __m256 j = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.318309886183790671f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(j, _mm256_set1_ps(3.14159274101257324f), x);
    r = _mm256_fnmadd_ps(j, _mm256_set1_ps(-8.74227800037248e-08f), r);
    const __m256 r2 = _mm256_mul_ps(r, r);
    __m256 p = _mm256_set1_ps(-2.50521083854417e-08f);
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(2.75573192239859e-06f));
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.98412698412698e-04f));
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(8.33333333333333e-03f));
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.66666666666667e-01f));
    p = _mm256_fmadd_ps(_mm256_mul_ps(p, r2), r, r);
    return _mm256_mul_ps(p, p);
}

void snake_avx2(const float * src, float * dst, const float * alpha, const float * inv_alpha, int64_t T, int64_t C) {
    parallel_for(T, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            const float * x = src + t * C;
            float * y = dst + t * C;
            for (int64_t c = 0; c < C; c += 8) {
                const __m256 v = _mm256_loadu_ps(x + c);
                const __m256 s = sin2_ps(_mm256_mul_ps(v, _mm256_loadu_ps(alpha + c)));
                _mm256_storeu_ps(y + c, _mm256_fmadd_ps(s, _mm256_loadu_ps(inv_alpha + c), v));
            }
        }
    });
}

}  // namespace dacn
