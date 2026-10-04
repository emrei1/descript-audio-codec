// Signal processing around the codec (see dsp.cpp).
#pragma once

#include "common.h"

namespace dacn {

// Integrated loudness (ITU-R BS.1770-4, as audiotools.Meter with IIR K-weighting) in LUFS,
// clamped below at -70 like AudioSignal.loudness(). Several channels are gated jointly with
// BS.1770 channel weights (1, 1, 1, 1.41, 1.41).
double loudness_lufs(const float * x, int64_t n, int sample_rate);
double loudness_lufs(const std::vector<const float *> & chans, int64_t n, int sample_rate);

// julius.resample_frac(x, old_sr, new_sr) (zeros=24, rolloff=0.945); length floor(n*new/old).
std::vector<float> resample_frac(const float * x, int64_t n, int old_sr, int new_sr);

// Linked look-ahead peak limiter over interleaved audio (ceiling linear, e.g. 0.966 = -0.3 dBFS).
// Returns the largest gain reduction applied in dB.
double limit_peaks(float * interleaved, int64_t frames, int channels, int sample_rate, float ceiling,
                   double lookahead_ms = 5.0, double release_ms = 100.0);

}  // namespace dacn
