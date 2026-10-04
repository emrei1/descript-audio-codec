// Per-platform locations: model file, cache directory, executable path.
#pragma once

#include "common.h"

namespace dacn {

fs::path executable_path();
// $DAC_MODEL, else dac.gguf next to the executable, else the bundle's Resources (macOS),
// else the per-user data directory.
fs::path default_model_path();
// Windows: %LOCALAPPDATA%\DacPlayer\cache   Linux: $XDG_CACHE_HOME/dac-player
// macOS: ~/Library/Caches/com.descript.dac-player
fs::path cache_dir();

}  // namespace dacn
