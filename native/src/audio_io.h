// Audio file input (WAV / FLAC / MP3 via dr_libs) and float32 WAV output.
#pragma once

#include "common.h"

namespace dacn {

struct audio_buffer {
    std::vector<float> samples;   // interleaved
    int64_t frames = 0;
    int channels = 0;
    int sample_rate = 0;
};

bool read_audio(const fs::path & path, audio_buffer & out, std::string & err);

// WAVE_FORMAT_IEEE_FLOAT (EXTENSIBLE for more than 2 channels).
bool write_wav_f32(const fs::path & path, const float * interleaved, int64_t frames, int channels, int sample_rate,
                   std::string & err);

}  // namespace dacn
