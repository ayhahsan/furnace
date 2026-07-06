# M3 checkpoint: compare furnace tokenization against HuggingFace AutoTokenizer
# for Qwen/Qwen2.5-0.5B-Instruct, and verify furnace round-trips every case.
# Usage: python check_m3.py <model.gguf> <path-to-furnace.exe> [tokenizer-src]
# tokenizer-src defaults to the HF hub id; pass a local directory containing
# tokenizer.json + tokenizer_config.json to run offline.

import subprocess
import sys

from transformers import AutoTokenizer

# Windows consoles default to a legacy codepage that cannot print the
# non-ASCII test strings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "नमस्ते, आप कैसे हैं?",  # Hindi
    "क्या आपने खाना खाया? मैं ठीक हूँ \U0001F44D",
    "mujhe kal office jaana hai yaar",  # Hinglish
    "chai peene chalein? office ke baad milte hain",
    'fn main() {\n    println!("hello");\n}',
    "def f(x):\n    return x**2\n\n",
    "In 2024, prices rose 3.14% across 1,000,000 items.",
    "a    b\t\tc\n\n\nd",
    "   leading and trailing   ",
    "I love pizza \U0001F355 and sushi \U0001F363!",
    "\U0001F914\U0001F914\U0001F914",
    "<|im_start|>user\nHello<|im_end|>",
    "<|endoftext|>",
    "Email me at test@example.com or visit https://example.com/path?q=1",
    "I'll say it's John's book, isn't it? We're DONE.",
    "",
]


def furnace_encode(exe, model, text):
    p = subprocess.run(
        [exe, "tokenize", model],
        input=text.encode("utf-8"), capture_output=True, check=True,
    )
    out = p.stdout.decode("utf-8").strip()
    return [int(x) for x in out.split()] if out else []


def furnace_decode(exe, model, ids):
    p = subprocess.run(
        [exe, "detokenize", model, "--ids", ",".join(map(str, ids))],
        capture_output=True, check=True,
    )
    out = p.stdout.decode("utf-8")
    assert out.endswith("\n")
    return out[:-1]  # exactly one trailing newline added by furnace


def report_divergence(case, ours, ref, tok):
    n = min(len(ours), len(ref))
    i = next((k for k in range(n) if ours[k] != ref[k]), n)
    print(f"  diverges at token index {i}")
    print(f"  furnace ids:   {ours[max(0, i - 2):i + 4]}")
    print(f"  reference ids: {ref[max(0, i - 2):i + 4]}")
    print(f"  reference tokens there: {tok.convert_ids_to_tokens(ref[max(0, i - 2):i + 4])}")
    consumed = len(tok.decode(ref[:i]).encode("utf-8"))
    raw = case.encode("utf-8")
    lo, hi = max(0, consumed - 12), consumed + 12
    print(f"  input bytes [{lo}:{hi}] around divergence: {raw[lo:hi]!r}")
    print(f"                 divergence at byte offset {consumed}")


def main():
    model, exe = sys.argv[1], sys.argv[2]
    src = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(src)

    for case in CASES:
        ref = tok(case, add_special_tokens=False)["input_ids"]
        ours = furnace_encode(exe, model, case)

        if ours != ref:
            print(f"FAIL encode {case!r}")
            report_divergence(case, ours, ref, tok)
            sys.exit(1)

        decoded = furnace_decode(exe, model, ours)
        if decoded != case:
            print(f"FAIL round-trip {case!r}")
            print(f"  got {decoded!r}")
            sys.exit(1)

        print(f"PASS {len(ref):4} tokens  {case!r}")

    print(f"\nall {len(CASES)} cases match HF and round-trip exactly")


if __name__ == "__main__":
    main()
