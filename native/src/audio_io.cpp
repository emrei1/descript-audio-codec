#include "audio_io.h"

#include <cstring>

#define DR_WAV_IMPLEMENTATION
#define DR_FLAC_IMPLEMENTATION
#define DR_MP3_IMPLEMENTATION
#include "dr_flac.h"
#include "dr_mp3.h"
#include "dr_wav.h"

namespace dacn {

bool read_audio(const fs::path & path, audio_buffer & out, std::string & err) {
    std::vector<uint8_t> bytes;
    if (!read_file(path, bytes, err)) return false;
    if (bytes.size() < 12) { err = "audio file too small"; return false; }
    const char * h = (const char *) bytes.data();

    out = audio_buffer{};
    if (!memcmp(h, "RIFF", 4) || !memcmp(h, "RIFX", 4) || !memcmp(h, "RF64", 4) || !memcmp(h, "riff", 4)) {
        drwav w;
        if (!drwav_init_memory(&w, bytes.data(), bytes.size(), nullptr)) { err = "unsupported WAV file"; return false; }
        out.channels = (int) w.channels;
        out.sample_rate = (int) w.sampleRate;
        out.samples.resize((size_t) (w.totalPCMFrameCount * w.channels));
        out.frames = (int64_t) drwav_read_pcm_frames_f32(&w, w.totalPCMFrameCount, out.samples.data());
        drwav_uninit(&w);
    } else if (!memcmp(h, "fLaC", 4)) {
        drflac * f = drflac_open_memory(bytes.data(), bytes.size(), nullptr);
        if (!f) { err = "unsupported FLAC file"; return false; }
        out.channels = (int) f->channels;
        out.sample_rate = (int) f->sampleRate;
        out.samples.resize((size_t) (f->totalPCMFrameCount * f->channels));
        out.frames = (int64_t) drflac_read_pcm_frames_f32(f, f->totalPCMFrameCount, out.samples.data());
        drflac_close(f);
    } else {
        drmp3 m;
        if (!drmp3_init_memory(&m, bytes.data(), bytes.size(), nullptr)) {
            err = "unsupported audio format (expected WAV, FLAC or MP3)";
            return false;
        }
        out.channels = (int) m.channels;
        out.sample_rate = (int) m.sampleRate;
        std::vector<float> chunk(4096 * (size_t) m.channels);
        drmp3_uint64 n;
        while ((n = drmp3_read_pcm_frames_f32(&m, 4096, chunk.data())) > 0)
            out.samples.insert(out.samples.end(), chunk.begin(), chunk.begin() + (size_t) (n * m.channels));
        out.frames = (int64_t) (out.samples.size() / m.channels);
        drmp3_uninit(&m);
    }
    out.samples.resize((size_t) (out.frames * out.channels));
    if (out.frames <= 0 || out.channels <= 0 || out.sample_rate <= 0) { err = "audio file has no samples"; return false; }
    return true;
}

bool write_wav_f32(const fs::path & path, const float * interleaved, int64_t frames, int channels, int sample_rate,
                   std::string & err) {
    const uint64_t data_bytes = (uint64_t) frames * channels * 4;
    if (data_bytes > 0xFFFFFF00ull) { err = "output too large for a WAV file"; return false; }
    FILE * f = open_file(path, "wb");
    if (!f) { err = "cannot create " + path.u8string(); return false; }
    auto w32 = [&](uint32_t v) { uint8_t b[4] = {(uint8_t) v, (uint8_t) (v >> 8), (uint8_t) (v >> 16), (uint8_t) (v >> 24)}; fwrite(b, 1, 4, f); };
    auto w16 = [&](uint16_t v) { uint8_t b[2] = {(uint8_t) v, (uint8_t) (v >> 8)}; fwrite(b, 1, 2, f); };
    const bool ext = channels > 2;
    const uint32_t fmt_size = ext ? 40 : 18;
    fwrite("RIFF", 1, 4, f); w32(4 + (8 + fmt_size) + 12 + 8 + (uint32_t) data_bytes); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(fmt_size);
    w16(ext ? 0xFFFE : 3);
    w16((uint16_t) channels); w32((uint32_t) sample_rate);
    w32((uint32_t) sample_rate * channels * 4); w16((uint16_t) (channels * 4)); w16(32);
    if (ext) {
        w16(22); w16(32);
        w32(channels >= 32 ? 0xFFFFFFFFu : (1u << channels) - 1);
        const uint8_t guid[16] = {0x03, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71};   // IEEE_FLOAT
        fwrite(guid, 1, 16, f);
    } else {
        w16(0);
    }
    fwrite("fact", 1, 4, f); w32(4); w32((uint32_t) frames);
    fwrite("data", 1, 4, f); w32((uint32_t) data_bytes);
    // samples are little-endian floats on every supported platform
    const size_t n = (size_t) frames * channels;
    const bool ok = fwrite(interleaved, 4, n, f) == n;
    if (fclose(f) != 0 || !ok) { err = "write failed (disk full?)"; return false; }
    return true;
}

}  // namespace dacn
