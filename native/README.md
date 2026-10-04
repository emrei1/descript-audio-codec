# Native CPU codec and desktop player

A dependency-light C++17 implementation of the Descript Audio Codec (44.1 kHz model): **encoder,
residual vector quantizer and decoder**, plus a desktop file handler so that double-clicking a
`.dac` file plays it in the system's media player on **Windows, Linux and macOS**.

It reproduces `DAC.compress()` / `DAC.decompress()` (chunked inference with `padding=False`,
loudness normalization, resampling) and reads and writes the regular `.dac` format, so files move
freely between this code and `python -m dac`.

## Results

Measured against the PyTorch reference in this repository (`tests/compare_with_python.py`):

| Check | Result |
|---|---|
| Encoder codes vs `DAC.compress` (mono, 48 kHz stereo, 96 kHz music) | **100 % identical** |
| Decoder vs `DAC.decompress` (`--reference` mode) | **SNR 103-115 dB** (float32 rounding) |
| `.dac` interop | Python loads/decodes native files and vice versa |

Speed on an Intel Core Ultra 7 255H laptop (12 threads, oneDNN backend):

| | Real-time factor | |
|---|---|---|
| Decoder (Windows / Linux) | 0.18-0.22 | ~5x faster than real time per channel |
| Encoder (Windows / Linux) | 0.09-0.11 | ~9-12x faster than real time per channel |
| PyTorch CPU decoder, same machine | ~1.0 | 30 s stereo song: 60.8 s vs ~15 s native |

## How it is fast

* **oneDNN direct convolutions** on x86-64 (JIT brgemm kernels for AVX2 / AVX-512): no im2col
  buffers; bias, residual addition (sum post-op written in place) and the final `tanh` are fused.
* **tap-GEMM backend** elsewhere: with channels-last activations a K-tap 1-D convolution is K SGEMM
  calls on strided views, so it maps directly onto Apple Accelerate (AMX on Apple silicon) or any
  CBLAS such as OpenBLAS, still without im2col.
* **Snake** in one pass: `sin^2` with an argument reduction by pi and an odd polynomial (AVX2
  version selected at run time; the portable loop auto-vectorizes on NEON).
* Weight norm and the quantizer's `out_proj` are folded at conversion time; weights are reordered
  once and execution plans are cached per chunk length.

## Build

```sh
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

oneDNN is downloaded and built automatically when it is not installed (x86-64 Windows/Linux).
macOS uses Accelerate; other CPUs use OpenBLAS if found. `-DDAC_BACKEND=GEMM|ONEDNN` and
`-DDAC_SGEMM=ACCELERATE|CBLAS|DNNL|REFERENCE` override the choice.

## Weights

```sh
pip install -e .                                   # the Python package, for the converter
python native/scripts/convert_weights.py -o dac.gguf   # downloads the 44 kHz checkpoint
```

The GGUF file holds float32 weights with weight norm folded. Its decoder part is compatible with
zonos2.cpp's `dac.gguf`.

## Command line

```sh
dac-native encode song.flac song.dac        # WAV / FLAC / MP3 input
dac-native decode song.dac song.wav         # player mode (see below); --reference = exact Python output
dac-native info song.dac
dac-native bench
```

## Desktop integration

| | Association | Progress UI | Playback |
|---|---|---|---|
| Windows | per-user registry ProgID `DacPlayer.dac` (HKCU, no admin) | Task dialog with Cancel | default `.wav` app (`ShellExecute`) |
| Linux | `audio/x-dac` shared-mime-info type + `dac-player.desktop` (`xdg-mime default`) | zenity, else notify-send | `xdg-open` |
| macOS | `com.descript.dac` exported UTI in `DAC Player.app` (`LSSetDefaultRoleHandlerForContentType`) | Cocoa window with Cancel | `NSWorkspace` (Music / QuickTime) |

Install scripts copy the program and model to a per-user location and register the association:

```sh
powershell -ExecutionPolicy Bypass -File native/packaging/install-windows.ps1 -BuildDir build -Model dac.gguf
sh native/packaging/install-linux.sh build dac.gguf
sh native/packaging/install-macos.sh build dac.gguf
```

Decoded audio is cached per file content (`%LOCALAPPDATA%\DacPlayer\cache`, `~/.cache/dac-player`,
`~/Library/Caches/com.descript.dac-player`) and pruned after 7 days, so re-opening a file is instant.

### Player mode vs reference mode

`DAC.decompress()` measures `input_db` on all channels together but normalizes each decoded channel
separately to it. For stereo this makes the output about 3 dB louder than the source and it can
clip (e.g. 27k clipped samples in a 30 s excerpt of a commercial master). The player therefore
measures loudness jointly (restoring the source level) and adds a 5 ms look-ahead limiter at
-0.3 dBFS. `dac-native decode --reference` reproduces the Python output exactly.

### Security

`.dac` files are pickles inside `.npy`, which `np.load(allow_pickle=True)` would execute. The
native reader never executes anything: it interprets only plain pickle data opcodes and the three
numpy constructors the format needs, and rejects everything else (a crafted file calling
`os.system` is refused with `refusing pickle global nt.system`).
