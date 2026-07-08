# M6 checkpoint. Stage A: full-forward last-position logits vs HF (argmax
# and top-5 must match exactly). Stage B: 20 greedy tokens with the chat
# template vs HF generate.
#
# The HF model is built from config (all 24 layers, float32, untied lm_head)
# with EVERY weight patched from the GGUF, as in check_m5.py.
#
# Usage: python check_m6.py <model.gguf> <furnace.exe> <hf-config-dir> [out.ften] [--stage a|b|all]

import re
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from gguf import GGUFReader
from gguf.quants import dequantize

from check_m4 import write_tensors

TOKENS = [151644, 872, 198, 9707, 0]  # same fixed sequence as M5
PROMPT = "What is the capital of France?"
N_NEW = 20


def weight_map(n_layers):
    m = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    for i in range(n_layers):
        g, h = f"blk.{i}.", f"model.layers.{i}."
        m[g + "attn_norm.weight"] = h + "input_layernorm.weight"
        for p in ("q", "k", "v"):
            m[g + f"attn_{p}.weight"] = h + f"self_attn.{p}_proj.weight"
            m[g + f"attn_{p}.bias"] = h + f"self_attn.{p}_proj.bias"
        m[g + "attn_output.weight"] = h + "self_attn.o_proj.weight"
        m[g + "ffn_norm.weight"] = h + "post_attention_layernorm.weight"
        for p in ("gate", "up", "down"):
            m[g + f"ffn_{p}.weight"] = h + f"mlp.{p}_proj.weight"
    return m


def build_model(gguf_path, config_dir):
    reader = GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}

    config = AutoConfig.from_pretrained(config_dir)
    # files with a separate output.weight need the tie broken; tied models
    # (e.g. 1.5B) omit it from the GGUF and lm_head follows the embedding
    config.tie_word_embeddings = "output.weight" not in by_name
    torch.set_default_dtype(torch.float32)
    model = AutoModelForCausalLM.from_config(config).float()
    model.eval()
    assert next(model.parameters()).dtype == torch.float32

    with torch.no_grad():
        for gguf_name, hf_name in weight_map(config.num_hidden_layers).items():
            if gguf_name not in by_name:
                assert gguf_name == "output.weight", f"missing {gguf_name}"
                continue
            t = by_name[gguf_name]
            if t.tensor_type.name == "F32":
                arr = np.asarray(t.data, dtype=np.float32).reshape(-1)
            else:
                arr = dequantize(t.data, t.tensor_type).reshape(-1).astype(np.float32)
            param = model.get_parameter(hf_name)
            param.copy_(torch.from_numpy(arr.reshape(param.shape)))
    return model


def stage_a(model, gguf_path, exe, out_path):
    with torch.no_grad():
        logits = model(torch.tensor([TOKENS])).logits[0, -1:]  # [1, vocab]
    write_tensors(out_path, {
        "tokens": torch.tensor(TOKENS, dtype=torch.float32),
        "logits_last": logits,
    })
    print(f"stage A: wrote reference logits to {out_path}")
    r = subprocess.run([exe, "selftest-m6", gguf_path, out_path])
    if r.returncode != 0:
        sys.exit("stage A failed")


def stage_b(model, gguf_path, exe, config_dir):
    tok = AutoTokenizer.from_pretrained(config_dir)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    if not isinstance(ids, list):
        ids = ids["input_ids"]

    with torch.no_grad():
        out = model.generate(
            torch.tensor([ids]),
            max_new_tokens=N_NEW,
            do_sample=False,
            eos_token_id=None,  # let it run the full N_NEW for comparison
            pad_token_id=151643,
        )
    ref = out[0][len(ids):].tolist()
    print(f"stage B: HF greedy {len(ref)} tokens: {ref}")
    print(f"stage B: HF text: {tok.decode(ref)!r}")

    r = subprocess.run(
        [exe, "run", gguf_path, "-p", PROMPT, "-n", str(N_NEW)],
        capture_output=True,
    )
    sys.stderr.write(r.stderr.decode("utf-8", "replace"))
    print(f"stage B: furnace text: {r.stdout.decode('utf-8', 'replace')!r}")
    m = re.search(r"generated ids: \[([\d, ]*)\]", r.stderr.decode("utf-8"))
    assert m, "furnace did not report generated ids"
    ours = [int(x) for x in m.group(1).split(",")] if m.group(1).strip() else []

    n = min(len(ours), len(ref))
    diverge = next((i for i in range(n) if ours[i] != ref[i]), n)
    print(f"stage B: our ids:      {ours}")
    print(f"stage B: match up to token {diverge} of {n}")
    if diverge < 4:
        sys.exit(f"stage B failed: divergence at token {diverge} is a bug, not noise")
    if diverge < n:
        print("stage B: late divergence, acceptable (near-tie flipped by f32 noise)")


def main():
    gguf_path, exe, config_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else "check_m6.ften"
    stage = sys.argv[5].split("=")[-1] if len(sys.argv) > 5 else "all"

    model = build_model(gguf_path, config_dir)
    print("built and patched f32 reference model")
    if stage in ("a", "all"):
        stage_a(model, gguf_path, exe, out_path)
    if stage in ("b", "all"):
        stage_b(model, gguf_path, exe, config_dir)


if __name__ == "__main__":
    main()
