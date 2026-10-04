#!/usr/bin/env sh
# Install "DAC Player.app" into ~/Applications and make it the default app for .dac files.
#   sh packaging/install-macos.sh [build-dir] [dac.gguf]      install
#   sh packaging/install-macos.sh --uninstall                 remove
set -eu
APP="$HOME/Applications/DAC Player.app"

if [ "${1:-}" = "--uninstall" ]; then
    [ -x "$APP/Contents/MacOS/DAC Player" ] && "$APP/Contents/MacOS/DAC Player" --unregister || true
    rm -rf "$APP" "$HOME/Library/Caches/com.descript.dac-player"
    echo "DAC Player was removed."
    exit 0
fi

BUILD="${1:-build}"
MODEL="${2:-dac.gguf}"
mkdir -p "$HOME/Applications"
rm -rf "$APP"
cp -R "$BUILD/DAC Player.app" "$APP"
mkdir -p "$APP/Contents/Resources"
cp "$MODEL" "$APP/Contents/Resources/dac.gguf"
cp "$BUILD/dac-native" "$APP/Contents/MacOS/dac-native"
# local builds are unsigned: sign ad hoc so Gatekeeper lets the bundle run on this Mac
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
"$APP/Contents/MacOS/DAC Player" --register
echo "DAC Player installed to $APP; .dac files now open with it."
