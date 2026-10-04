// C++ ports of dac.DAC.compress() and dac.DAC.decompress().
#pragma once

#include "audio_io.h"
#include "dacfile.h"
#include "engine.h"

#include <atomic>

namespace dacn {

struct progress {
    std::atomic<int64_t> done{0}, total{0};   // processed chunks
    std::atomic<bool> cancel{false};
};

struct compress_options {
    double win_duration = 5.0;   // seconds per chunk (python -m dac encode default)
    double normalize_db = -16.0;
    int n_quantizers = 0;        // 0 = all codebooks
};

struct decompress_options {
    // true : bit-for-bit dac.DAC.decompress, including its quirk of normalizing each channel
    //        separately to the joint input_db (stereo comes out ~3 dB hot and can clip).
    // false: joint BS.1770 loudness across channels (restores the source level) followed by a
    //        look-ahead peak limiter at `ceiling`.
    bool reference = false;
    float ceiling = 0.966f;   // -0.3 dBFS
};

struct decompress_result {
    audio_buffer audio;
    double neural_seconds = 0;   // time in the decoder network
    double limiter_db = 0;
};

bool compress(engine & eng, const audio_buffer & in, dac_file & out, std::string & err,
              const compress_options & opt = {}, progress * prog = nullptr);

bool decompress(engine & eng, const dac_file & f, decompress_result & out, std::string & err,
                const decompress_options & opt = {}, progress * prog = nullptr);

// Model geometry helpers mirroring CodecMixin.get_output_length / get_delay (unpadded convs).
int64_t codec_output_length(const model & m, int64_t input_length);
int64_t codec_delay(const model & m);

}  // namespace dacn
