# M2 checkpoint: compare furnace dequantization against the gguf package.
# Usage: python check_m2.py <model.gguf> <path-to-furnace.exe>

import random
import subprocess
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

TENSORS = ["blk.0.attn_q.weight", "output_norm.weight"]
COUNT = 100
TOLERANCE = 1e-5


def furnace_values(exe, model, name, offset, count):
    out = subprocess.run(
        [exe, "dump-tensor", model, name,
         "--offset", str(offset), "--count", str(count)],
        capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout
    values = np.array([float(x) for x in out.split()], dtype=np.float32)
    assert len(values) == count, f"expected {count} values, got {len(values)}"
    return values


def reference_values(reader, name):
    t = next(t for t in reader.tensors if t.name == name)
    if t.tensor_type.name == "F32":
        return np.asarray(t.data, dtype=np.float32).reshape(-1)
    return dequantize(t.data, t.tensor_type).reshape(-1).astype(np.float32)


def main():
    model, exe = sys.argv[1], sys.argv[2]
    reader = GGUFReader(model)
    random.seed(0)

    failures = 0
    for name in TENSORS:
        ref = reference_values(reader, name)
        n = len(ref)
        # first 100, plus a window at a random offset in the middle half
        offsets = [0, random.randint(n // 4, 3 * n // 4 - COUNT)]
        for offset in offsets:
            count = min(COUNT, n - offset)
            got = furnace_values(exe, model, name, offset, count)
            max_diff = float(np.abs(got - ref[offset:offset + count]).max())
            status = "PASS" if max_diff <= TOLERANCE else "FAIL"
            if status == "FAIL":
                failures += 1
            print(f"{status} {name} [{offset}:{offset + count}] "
                  f"max abs diff {max_diff:.3e}")

    if failures:
        print(f"{failures} comparisons failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
