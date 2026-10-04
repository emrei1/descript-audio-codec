// Descript Audio Codec ".dac" files (dac.DACFile, format version 1.0.0).
//
// A .dac file is `np.save` of a Python dict - a .npy object array whose payload is a pickle.
// Python opens it with np.load(allow_pickle=True), which can execute code from an untrusted
// file. The reader here interprets the pickle with a small stack machine that only knows plain
// data opcodes plus three whitelisted numpy constructors (ndarray / _reconstruct / dtype) and
// never calls anything else. The writer emits the same structure numpy produces, so files go
// both ways between this codec and `python -m dac`.
#pragma once

#include "common.h"

namespace dacn {

struct dac_file {
    // codes[(b * n_codebooks + k) * T + t]; b = channel (batch item), k = codebook, t = frame
    std::vector<int32_t> codes;
    int64_t B = 0, n_codebooks = 0, T = 0;

    int64_t chunk_length = 0;
    int64_t original_length = 0;   // samples per channel at sample_rate
    double input_db = 0;           // loudness of the source (LUFS); decode is normalized back to it
    int64_t channels = 1;
    int64_t sample_rate = 44100;   // rate of the ORIGINAL audio (the codec itself runs at the model rate)
    bool padding = true;
    std::string dac_version = "1.0.0";

    const int32_t * code_row(int64_t b, int64_t k) const { return codes.data() + (b * n_codebooks + k) * T; }
};

bool dac_file_load(const fs::path & path, dac_file & out, std::string & err);
bool dac_file_parse(const uint8_t * data, size_t size, dac_file & out, std::string & err);

bool dac_file_save(const fs::path & path, const dac_file & f, std::string & err);
std::vector<uint8_t> dac_file_serialize(const dac_file & f);

}  // namespace dacn
