#!/usr/bin/env python3
"""Convert a DAC checkpoint (.pth) into the GGUF file used by the native CPU codec.

    python native/scripts/convert_weights.py [weights.pth] -o dac.gguf

With no checkpoint argument the default 44 kHz model is downloaded via `dac.utils.download()`.

Everything is stored as float32 with weight norm folded in (w = g * v / ||v||). Tensor names:

  enc.conv_in.*                         Conv1d(1 -> d, k7)
  enc.b{B}.res{R}.{snake1,conv1,snake2,conv2}.*   residual units (k7 dilation 1/3/9, then k1)
  enc.b{B}.snake.* / enc.b{B}.conv.*    Snake + strided Conv1d(k = 2*stride)
  enc.snake_out.* / enc.conv_out.*      Snake + Conv1d(k3) to the latent
  quant.{i}.in_w / in_b                 in_proj (latent -> codebook_dim), 1x1 conv as a matrix
  quant.{i}.codebook                    [codebook_size, codebook_dim] raw codebook
  quant.{i}.out_b                       out_proj bias
  quant.{i}.table                       codebook @ out_proj^T, i.e. out_proj folded into a lookup
                                        table [codebook_size, latent] (bias excluded)
  quant.bias                            sum of all out_proj biases (decode path)
  dec.conv_in.* / dec.b{B}.{snake,convt}.* / dec.b{B}.res{R}.* / dec.snake_out.* / dec.conv_out.*

The decoder half uses the same names and layout as zonos2.cpp's dac.gguf, so either file
works for decoding. Conv weights keep PyTorch's [OC, IC, K] (ConvTranspose: [IC, OC, K]);
GGUF lists dimensions innermost-first, so they appear as ne = [K, IC, OC].
"""
import argparse
import struct

import numpy as np
import torch

GGUF_MAGIC = 0x46554747
ALIGN = 32
T_U32, T_STR, T_ARR, T_I32 = 4, 8, 9, 5


def eff_weight(sd, prefix):
    g = sd[prefix + ".weight_g"]
    v = sd[prefix + ".weight_v"]
    return torch._weight_norm(v, g, 0).detach().float().numpy()


class GGUFWriter:
    """Minimal GGUF v3 writer (F32 tensors, u32 / i32-array / string metadata)."""

    def __init__(self):
        self.kv = []
        self.tensors = []

    def u32(self, key, v):
        self.kv.append((key, T_U32, struct.pack("<I", int(v))))

    def string(self, key, v):
        b = v.encode()
        self.kv.append((key, T_STR, struct.pack("<Q", len(b)) + b))

    def i32_array(self, key, vals):
        self.kv.append((key, T_ARR, struct.pack("<IQ", T_I32, len(vals)) + struct.pack(f"<{len(vals)}i", *vals)))

    def tensor(self, name, arr):
        self.tensors.append((name, np.ascontiguousarray(arr, dtype=np.float32)))

    def write(self, path):
        def s(x):
            b = x.encode()
            return struct.pack("<Q", len(b)) + b

        head = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(self.tensors), len(self.kv) + 1)
        head += s("general.alignment") + struct.pack("<II", T_U32, ALIGN)
        for key, t, payload in self.kv:
            head += s(key) + struct.pack("<I", t) + payload
        offset = 0
        for name, a in self.tensors:
            dims = a.shape[::-1]
            head += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
            head += struct.pack("<IQ", 0, offset)
            offset += (a.nbytes + ALIGN - 1) // ALIGN * ALIGN
        with open(path, "wb") as f:
            f.write(head)
            f.write(b"\0" * ((-len(head)) % ALIGN))
            for _, a in self.tensors:
                f.write(a.tobytes())
                f.write(b"\0" * ((-a.nbytes) % ALIGN))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", nargs="?", help="DAC .pth checkpoint (default: download the 44 kHz model)")
    ap.add_argument("--model_type", default="44khz")
    ap.add_argument("--model_bitrate", default="8kbps")
    ap.add_argument("-o", "--out", default="dac.gguf")
    args = ap.parse_args()

    path = args.ckpt
    if path is None:
        import dac

        path = dac.utils.download(model_type=args.model_type, model_bitrate=args.model_bitrate)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, kw = ck["state_dict"], ck["metadata"]["kwargs"]

    enc_dim = kw.get("encoder_dim", 64)
    enc_rates = list(kw.get("encoder_rates", [2, 4, 8, 8]))
    dec_dim = kw.get("decoder_dim", 1536)
    dec_rates = list(kw.get("decoder_rates", [8, 8, 4, 2]))
    latent = kw.get("latent_dim") or enc_dim * 2 ** len(enc_rates)
    n_cb = kw.get("n_codebooks", 9)
    cb_size = kw.get("codebook_size", 1024)
    cb_dim = kw.get("codebook_dim", 8)
    sr = kw.get("sample_rate", 44100)

    w = GGUFWriter()
    w.string("general.name", f"DAC-{args.model_type}-{args.model_bitrate}")
    for k, v in [("n_codebooks", n_cb), ("codebook_size", cb_size), ("codebook_dim", cb_dim), ("latent_dim", latent),
                 ("encoder_dim", enc_dim), ("decoder_dim", dec_dim), ("sample_rate", sr),
                 ("hop_length", int(np.prod(enc_rates))), ("audio_pad_id", cb_size + 1)]:
        w.u32(f"dac.{k}", v)
    w.i32_array("dac.encoder_rates", enc_rates)
    w.i32_array("dac.decoder_rates", dec_rates)

    def conv(name, prefix):
        w.tensor(name + ".weight", eff_weight(sd, prefix))
        w.tensor(name + ".bias", sd[prefix + ".bias"].float().numpy()[:, None])

    def snake(name, prefix):
        a = sd[prefix + ".alpha"].float().numpy().reshape(-1, 1)
        w.tensor(name + ".alpha", a)
        w.tensor(name + ".inv_alpha", 1.0 / (a + 1e-9))

    def res_units(name, prefix, first_index):
        for r in range(3):
            ru = f"{prefix}.block.{first_index + r}"
            snake(f"{name}.res{r}.snake1", f"{ru}.block.0")
            conv(f"{name}.res{r}.conv1", f"{ru}.block.1")
            snake(f"{name}.res{r}.snake2", f"{ru}.block.2")
            conv(f"{name}.res{r}.conv2", f"{ru}.block.3")

    # ---- encoder: block.0 conv_in, block.1..N EncoderBlock, then Snake + conv_out ----
    conv("enc.conv_in", "encoder.block.0")
    for b in range(len(enc_rates)):
        m = f"encoder.block.{b + 1}"
        res_units(f"enc.b{b}", m, 0)
        snake(f"enc.b{b}.snake", f"{m}.block.3")
        conv(f"enc.b{b}.conv", f"{m}.block.4")
    n = len(enc_rates) + 1
    snake("enc.snake_out", f"encoder.block.{n}")
    conv("enc.conv_out", f"encoder.block.{n + 1}")

    # ---- quantizer ----
    total_bias = np.zeros(latent, dtype=np.float64)
    for i in range(n_cb):
        q = f"quantizer.quantizers.{i}"
        w_in = eff_weight(sd, f"{q}.in_proj")[:, :, 0]               # [cb_dim, latent]
        w_out = eff_weight(sd, f"{q}.out_proj")[:, :, 0]             # [latent, cb_dim]
        cb = sd[f"{q}.codebook.weight"].float().numpy()               # [cb_size, cb_dim]
        b_out = sd[f"{q}.out_proj.bias"].float().numpy()
        w.tensor(f"quant.{i}.in_w", w_in)
        w.tensor(f"quant.{i}.in_b", sd[f"{q}.in_proj.bias"].float().numpy())
        w.tensor(f"quant.{i}.codebook", cb)
        w.tensor(f"quant.{i}.out_b", b_out)
        w.tensor(f"quant.{i}.table", cb @ w_out.T)                    # [cb_size, latent]
        total_bias += b_out
    w.tensor("quant.bias", total_bias)

    # ---- decoder: model.0 conv_in, model.1..N DecoderBlock, then Snake + conv_out ----
    conv("dec.conv_in", "decoder.model.0")
    for b in range(len(dec_rates)):
        m = f"decoder.model.{b + 1}"
        snake(f"dec.b{b}.snake", f"{m}.block.0")
        conv(f"dec.b{b}.convt", f"{m}.block.1")
        res_units(f"dec.b{b}", m, 2)
    n = len(dec_rates) + 1
    snake("dec.snake_out", f"decoder.model.{n}")
    conv("dec.conv_out", f"decoder.model.{n + 1}")

    w.write(args.out)
    print(f"wrote {args.out}: {len(w.tensors)} tensors, sample_rate={sr}, latent={latent}, "
          f"encoder_rates={enc_rates}, decoder_rates={dec_rates}, codebooks={n_cb}x{cb_size}")


if __name__ == "__main__":
    main()
