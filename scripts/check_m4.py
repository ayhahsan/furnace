# M4 checkpoint: generate reference inputs/outputs with PyTorch (fixed seed),
# write them to an FTEN tensor file, and run `furnace selftest-m4` on it.
# Usage: python check_m4.py <path-to-furnace.exe> [out.ften]
#
# FTEN format (little-endian):
#   magic "FTEN" | count u32
#   per tensor: name_len u32, name, n_dims u32, dims u64[n_dims] (row-major
#   shape), f32 data

import struct
import subprocess
import sys

import torch


def write_tensors(path, tensors):
    with open(path, "wb") as f:
        f.write(b"FTEN")
        f.write(struct.pack("<I", len(tensors)))
        for name, t in tensors.items():
            t = t.contiguous().float()
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", t.dim()))
            for d in t.shape:
                f.write(struct.pack("<Q", d))
            f.write(t.numpy().tobytes())  # f32 little-endian on x86


def main():
    exe = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "check_m4.ften"

    torch.manual_seed(0)
    seq, d_in, d_out = 4, 8, 6
    eps = 1e-6

    a = torch.randn(seq, d_in)
    w = torch.randn(d_out, d_in)          # [out, in], PyTorch Linear layout
    x = torch.randn(seq, d_in)
    b = torch.randn(seq, d_in)
    norm_w = torch.randn(d_in)
    sm_in = torch.randn(seq, d_in) * 30   # spread logits to stress stability
    gate = torch.randn(seq, d_in)
    up = torch.randn(seq, d_in)

    tensors = {
        "eps": torch.tensor([eps]),
        "a": a, "w": w, "x": x, "b": b, "norm_w": norm_w,
        "sm_in": sm_in, "gate": gate, "up": up,
        "matmul_out": a @ w.T,
        "rmsnorm_out": x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * norm_w,
        "softmax_out": torch.softmax(sm_in, dim=-1),
        "swiglu_out": torch.nn.functional.silu(gate) * up,
        "add_out": x + b,
        "mul_out": x * b,
    }
    write_tensors(out_path, tensors)

    result = subprocess.run([exe, "selftest-m4", out_path])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
