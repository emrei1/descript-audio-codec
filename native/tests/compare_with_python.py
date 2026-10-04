#!/usr/bin/env python3
"""Check the native codec against the PyTorch reference.

    python native/tests/compare_with_python.py --native build/dac-native --model dac.gguf [--audio in.wav]

For each test signal it verifies, against `dac.DAC.compress` / `decompress`:
  1. encode:  share of identical codes between native and PyTorch encodes
  2. decode:  native decode (--reference) of the PyTorch .dac vs PyTorch decompress (SNR)
  3. interop: PyTorch can load and decompress a .dac written by the native encoder
Without --audio a deterministic synthetic music-like signal is used (CI has no audio files).
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
from audiotools import AudioSignal

import dac


def synth(sr, seconds, channels, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    out = []
    for c in range(channels):
        x = np.zeros_like(t)
        for f0 in rng.uniform(80, 900, 6):           # a few decaying harmonic notes
            onset = rng.uniform(0, seconds * 0.8)
            env = np.where(t > onset, np.exp(-(t - onset) * rng.uniform(1, 4)), 0)
            for h in range(1, 6):
                x += env * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) / h
        x += 0.02 * rng.standard_normal(len(t))       # noise floor
        out.append(x / np.abs(x).max() * 0.6)
    return np.stack(out, 1).astype(np.float32)


def snr(ref, test):
    n = min(len(ref), len(test))
    e = ref[:n] - test[:n]
    return 10 * np.log10((ref[:n] ** 2).mean() / max((e ** 2).mean(), 1e-30))


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"command failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", action="append", default=[])
    ap.add_argument("--min-code-match", type=float, default=0.97)
    ap.add_argument("--min-snr", type=float, default=60.0)
    args = ap.parse_args()
    args.native = os.path.abspath(args.native)
    args.model = os.path.abspath(args.model)

    torch.set_num_threads(os.cpu_count() or 4)
    model = dac.DAC.load(dac.utils.download(model_type="44khz")).eval()
    tmp = tempfile.mkdtemp()
    cases = [(p, None) for p in args.audio]
    if not cases:
        for name, sr, sec, ch in [("short_mono_44k", 44100, 0.8, 1), ("stereo_48k_12s", 48000, 12.0, 2)]:
            path = os.path.join(tmp, name + ".wav")
            sf.write(path, synth(sr, sec, ch), sr, subtype="FLOAT")
            cases.append((path, name))

    failed = False
    for path, name in cases:
        name = name or os.path.splitext(os.path.basename(path))[0]
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        sig = AudioSignal(torch.from_numpy(y.T.copy())[None], sr)
        py_dac = os.path.join(tmp, name + "_py.dac")
        f_py = model.compress(sig, win_duration=5.0)
        f_py.save(py_dac)
        ref = model.decompress(dac.DACFile.load(py_dac)).audio_data[0].numpy().T

        nat_dac = os.path.join(tmp, name + "_native.dac")
        run([args.native, "encode", path, nat_dac, "--model", args.model])
        f_nat = dac.DACFile.load(nat_dac)                                   # interop: Python reads native file
        same_shape = tuple(f_nat.codes.shape) == tuple(f_py.codes.shape)
        match = float((f_nat.codes == f_py.codes).float().mean()) if same_shape else 0.0
        meta_ok = all(getattr(f_nat, k) == getattr(f_py, k) for k in
                      ["chunk_length", "original_length", "channels", "sample_rate", "padding"])
        db_diff = abs(float(f_nat.input_db) - float(f_py.input_db))
        py_from_native = model.decompress(f_nat).audio_data[0].numpy().T   # Python decodes native codes

        nat_wav = os.path.join(tmp, name + "_native.wav")
        run([args.native, "decode", py_dac, nat_wav, "--reference", "--model", args.model])
        nat = sf.read(nat_wav, dtype="float32", always_2d=True)[0]
        dec_snr = snr(ref, nat)

        ok = same_shape and meta_ok and match >= args.min_code_match and dec_snr >= args.min_snr and db_diff < 0.05
        failed |= not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: codes {tuple(f_nat.codes.shape)} match={match * 100:.2f}% "
              f"metadata={'ok' if meta_ok else 'MISMATCH'} input_db diff={db_diff:.4f} | "
              f"native decode vs PyTorch SNR={dec_snr:.1f} dB | PyTorch decoded native .dac: "
              f"{py_from_native.shape[0]} frames")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
