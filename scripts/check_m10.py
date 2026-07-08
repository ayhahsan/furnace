# M10 checkpoint: for every quantized dtype present in a GGUF file, pick one
# tensor and compare furnace dump-tensor output against gguf.quants.dequantize
# at the start and at a random mid-tensor offset (the M2 pattern).
# Usage: python check_m10.py <model.gguf> <path-to-furnace.exe>

import random
import subprocess
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

COUNT = 100
TOLERANCE = 1e-5


def furnace_values(exe, model, name, offset, count):
    out = subprocess.run(
        [exe, "dump-tensor", model, name,
         "--offset", str(offset), "--count", str(count)],
        capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout
    return np.array([float(x) for x in out.split()], dtype=np.float32)


def main():
    model, exe = sys.argv[1], sys.argv[2]
    reader = GGUFReader(model)
    random.seed(0)

    by_dtype = {}
    for t in reader.tensors:
        by_dtype.setdefault(t.tensor_type.name, t)
    print("dtypes present:", sorted(by_dtype))

    failures = 0
    for dtype, t in sorted(by_dtype.items()):
        if dtype == "F32":
            ref = np.asarray(t.data, dtype=np.float32).reshape(-1)
        else:
            ref = dequantize(t.data, t.tensor_type).reshape(-1).astype(np.float32)
        n = len(ref)
        offsets = [0, random.randint(n // 4, max(n // 4 + 1, 3 * n // 4 - COUNT))]
        for offset in offsets:
            count = min(COUNT, n - offset)
            got = furnace_values(exe, model, t.name, offset, count)
            max_diff = float(np.abs(got - ref[offset:offset + count]).max())
            ok = max_diff <= TOLERANCE
            failures += 0 if ok else 1
            print(f"{'PASS' if ok else 'FAIL'} {dtype:6} {t.name} "
                  f"[{offset}:{offset + count}] max abs diff {max_diff:.3e}")

    if failures:
        sys.exit(f"{failures} comparisons failed")


if __name__ == "__main__":
    main()
