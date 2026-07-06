# M1 checkpoint: compare `furnace inspect` output against the official gguf
# Python package. Usage: python check_m1.py <model.gguf> <path-to-furnace.exe>

import re
import subprocess
import sys

from gguf import GGUFReader

TENSOR_LINE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+\[([\d, ]*)\]\s+(\w+)\s+offset (\d+)$"
)


def read_reference(model_path):
    reader = GGUFReader(model_path)
    # ReaderTensor.data_offset is absolute; header offsets are relative to the
    # data section. The first tensor sits at relative 0, so the smallest
    # absolute offset IS the data section start.
    data_start = min(t.data_offset for t in reader.tensors)
    ref = {}
    for t in reader.tensors:
        dims = tuple(int(d) for d in t.shape)  # ggml order, matches furnace
        ref[t.name] = (dims, t.tensor_type.name, t.data_offset - data_start)
    return ref, data_start


def read_furnace(model_path, exe_path):
    out = subprocess.run(
        [exe_path, "inspect", model_path],
        capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout
    got = {}
    data_start = None
    for line in out.splitlines():
        m = re.search(r"data starts at byte (\d+)", line)
        if m:
            data_start = int(m.group(1))
        m = TENSOR_LINE.match(line)
        if m:
            _, name, dims, dtype, offset = m.groups()
            dims = tuple(int(d) for d in dims.split(",")) if dims.strip() else ()
            got[name] = (dims, dtype, int(offset))
    return got, data_start


def main():
    model_path, exe_path = sys.argv[1], sys.argv[2]
    ref, ref_start = read_reference(model_path)
    got, got_start = read_furnace(model_path, exe_path)

    failures = 0
    if got_start != ref_start:
        print(f"FAIL data_start: furnace={got_start} reference={ref_start}")
        failures += 1
    if set(got) != set(ref):
        print(f"FAIL tensor names: missing={set(ref) - set(got)} extra={set(got) - set(ref)}")
        failures += 1
    for name in sorted(set(got) & set(ref)):
        if got[name] != ref[name]:
            print(f"FAIL {name}: furnace={got[name]} reference={ref[name]}")
            failures += 1

    if failures == 0:
        print(f"PASS: {len(ref)} tensors match (name, dims, dtype, offset), "
              f"data_start={ref_start}")
    else:
        print(f"{failures} mismatches")
        sys.exit(1)


if __name__ == "__main__":
    main()
