#!/usr/bin/env sh
# Install DAC Player for the current user and associate .dac files with it.
#   sh packaging/install-linux.sh [build-dir] [dac.gguf]      install
#   sh packaging/install-linux.sh --uninstall                 remove
set -eu
PREFIX="${XDG_DATA_HOME:-$HOME/.local/share}/dac-player"
BIN="$HOME/.local/bin"

if [ "${1:-}" = "--uninstall" ]; then
    [ -x "$PREFIX/dac-player" ] && "$PREFIX/dac-player" --unregister || true
    rm -rf "$PREFIX" "${XDG_CACHE_HOME:-$HOME/.cache}/dac-player" "$BIN/dac-native"
    echo "DAC Player was removed."
    exit 0
fi

BUILD="${1:-build}"
MODEL="${2:-dac.gguf}"
mkdir -p "$PREFIX" "$BIN"
install -m 755 "$BUILD/dac-player" "$BUILD/dac-native" "$PREFIX/"
install -m 644 "$MODEL" "$PREFIX/dac.gguf"
ln -sf "$PREFIX/dac-native" "$BIN/dac-native"
"$PREFIX/dac-player" --register
echo "DAC Player installed to $PREFIX; .dac files now open with it."
