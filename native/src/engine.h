// DAC encoder / decoder graphs on top of a convolution backend.
//
// Both run in either of DAC's two modes:
//   padded   (model.padding = True, used for audio shorter than the window): PyTorch padding,
//            encoder maps N samples -> N / hop frames, decoder L frames -> L * hop samples.
//   unpadded (model.padding = False, chunked compress/decompress): every conv unpadded and the
//            residual branch center-cropped, exactly like the reference implementation.
// Execution plans (buffers, primitives, packed weights) are built once per (length, mode).
#pragma once

#include "backend.h"
#include "model.h"

#include <map>

namespace dacn {

class engine {
public:
    explicit engine(const model & m);
    ~engine();

    const model & weights() const { return m_; }
    const char * backend_name() const;

    // audio: N mono samples at the model rate -> codes[k * frames + t] for k < n_codebooks.
    void encode(const float * audio, int64_t N, bool padded, std::vector<int32_t> & codes, int64_t & frames);
    // codes[k] points at L indices for codebook k (n_cb <= n_codebooks) -> waveform.
    void decode(const int32_t * const * codes, int n_cb, int64_t L, bool padded, std::vector<float> & out);

    int64_t encoder_out_len(int64_t N, bool padded) const;
    int64_t decoder_out_len(int64_t L, bool padded) const;

private:
    struct plan;
    plan & enc_plan(int64_t N, bool padded);
    plan & dec_plan(int64_t L, bool padded);

    const model & m_;
    std::unique_ptr<backend> be_;
    std::map<std::pair<int64_t, bool>, std::unique_ptr<plan>> enc_, dec_;
};

}  // namespace dacn
