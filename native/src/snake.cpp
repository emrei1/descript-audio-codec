// Snake activation: x + inv_alpha[c] * sin(alpha[c] * x)^2.
// sin^2 ignores the sign of sin, so x is reduced by multiples of pi into [-pi/2, pi/2]
// (two-term Cody-Waite) and the odd Taylor series is evaluated to r^11 (error < 6e-8).
// Portable loop below (auto-vectorizes on NEON / SSE); an AVX2+FMA version is picked at run time.
#include "backend.h"

#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#define DAC_X86 1
#ifdef _MSC_VER
#include <intrin.h>
#endif
#endif

namespace dacn {

#ifdef DAC_X86
void snake_avx2(const float * src, float * dst, const float * alpha, const float * inv_alpha, int64_t T, int64_t C);

static bool cpu_has_avx2_fma() {
#ifdef _MSC_VER
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!(fma && osxsave && avx) || (_xgetbv(0) & 6) != 6) return false;
    __cpuidex(r, 7, 0);
    return (r[1] >> 5) & 1;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}
#endif

static inline float sin2_poly(float x) {
    const float j = std::nearbyint(x * 0.318309886183790671f);
    float r = x - j * 3.14159274101257324f;
    r = r - j * -8.74227800037248e-08f;
    const float r2 = r * r;
    float p = -2.50521083854417e-08f;
    p = p * r2 + 2.75573192239859e-06f;
    p = p * r2 - 1.98412698412698e-04f;
    p = p * r2 + 8.33333333333333e-03f;
    p = p * r2 - 1.66666666666667e-01f;
    p = (p * r2) * r + r;
    return p * p;
}

static void snake_portable(const float * src, float * dst, const float * alpha, const float * inv_alpha, int64_t T,
                           int64_t C) {
    parallel_for(T, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            const float * x = src + t * C;
            float * y = dst + t * C;
            for (int64_t c = 0; c < C; ++c) y[c] = x[c] + inv_alpha[c] * sin2_poly(alpha[c] * x[c]);
        }
    });
}

void snake(const float * src, float * dst, const float * alpha, const float * inv_alpha, int64_t T, int64_t C) {
#ifdef DAC_X86
    static const bool avx2 = cpu_has_avx2_fma();
    if (avx2 && C % 8 == 0) { snake_avx2(src, dst, alpha, inv_alpha, T, C); return; }
#endif
    snake_portable(src, dst, alpha, inv_alpha, T, C);
}

}  // namespace dacn
