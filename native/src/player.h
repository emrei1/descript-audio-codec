// Platform-independent part of the desktop .dac player: a background decode job and the WAV cache.
// The platform front ends (player_win.cpp, player_linux.cpp, player_mac.mm) handle file
// association, the progress UI and handing the result to the default media player.
#pragma once

#include "codec.h"

#include <chrono>

namespace dacn {

constexpr const char * kAppName = "DAC Player";
constexpr const char * kMimeType = "audio/x-dac";
constexpr const char * kUti = "com.descript.dac";

// Decoded WAV for `input` in the per-user cache: <cache>/<stem>-<content hash>.wav. Also drops
// cache entries unused for 7 days.
fs::path cached_wav_for(const fs::path & input);

class player_job {
public:
    player_job(fs::path input, fs::path wav_out);

    void run();                       // blocking; call on a worker thread
    void cancel() { prog_.cancel = true; }

    bool finished() const { return finished_; }
    bool ok() const { return ok_; }
    bool cancelled() const { return err_ == "cancelled"; }
    const std::string & error_text() const { return err_; }
    double fraction() const;          // 0..1
    std::string status() const;       // one line of human-readable progress
    std::string title() const;

private:
    fs::path in_, out_;
    progress prog_;
    std::atomic<int> stage_{0};       // 0 loading, 1 decoding, 2 writing, 3 done
    std::atomic<bool> finished_{false};
    bool ok_ = false;
    std::string err_;
    double audio_seconds_ = 0;
    std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now(), t_decode_;
};

}  // namespace dacn
