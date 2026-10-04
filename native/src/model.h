// DAC weights + hyper-parameters, loaded from the GGUF written by scripts/convert_weights.py
// (a decoder-only zonos2.cpp dac.gguf also works for decoding).
#pragma once

#include "common.h"

#include <map>
#include <string>
#include <vector>

namespace dacn {

struct tensor {
    std::vector<int64_t> ne;   // GGUF order: ne[0] innermost
    const float * data = nullptr;
    size_t count = 0;
};

class model {
public:
    bool load(const fs::path & path, std::string & err);

    const tensor & get(const std::string & name) const;   // throws dacn::error if missing
    bool has(const std::string & name) const { return tensors_.count(name) != 0; }
    bool has_encoder() const { return has("enc.conv_in.weight") && has("quant.0.in_w"); }

    int sample_rate = 44100, n_codebooks = 9, codebook_size = 1024, codebook_dim = 8;
    int latent_dim = 1024, encoder_dim = 64, decoder_dim = 1536, hop_length = 512;
    std::vector<int> encoder_rates{2, 4, 8, 8}, decoder_rates{8, 8, 4, 2};

private:
    fbuf buf_;
    std::map<std::string, tensor> tensors_;
};

}  // namespace dacn
