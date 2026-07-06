# M5 checkpoint: dump every intermediate of Qwen2.5-0.5B block 0 from HF
# transformers to an FTEN file, then run `furnace selftest-m5` against it.
#
# The HF model is built from config (1 layer, float32) and its embedding and
# block-0 weights are PATCHED with the dequantized GGUF tensors, so both
# sides compute from identical weights: any diff is a computation bug, not
# Q8_0 quantization noise.
#
# Usage: python check_m5.py <model.gguf> <furnace.exe|-> <hf-config-dir> [out.ften]
#        pass "-" as the exe to only write the reference file.

import sys

import numpy as np
import torch
import transformers.models.qwen2.modeling_qwen2 as qwen_mod
from transformers import AutoConfig, AutoModelForCausalLM
from gguf import GGUFReader
from gguf.quants import dequantize

from check_m4 import write_tensors

TOKENS = [151644, 872, 198, 9707, 0]  # <|im_start|> user \n Hello !

# GGUF tensor name -> HF parameter name (module prefix "model." added below)
WEIGHT_MAP = {
    "token_embd.weight": "embed_tokens.weight",
    "blk.0.attn_norm.weight": "layers.0.input_layernorm.weight",
    "blk.0.attn_q.weight": "layers.0.self_attn.q_proj.weight",
    "blk.0.attn_q.bias": "layers.0.self_attn.q_proj.bias",
    "blk.0.attn_k.weight": "layers.0.self_attn.k_proj.weight",
    "blk.0.attn_k.bias": "layers.0.self_attn.k_proj.bias",
    "blk.0.attn_v.weight": "layers.0.self_attn.v_proj.weight",
    "blk.0.attn_v.bias": "layers.0.self_attn.v_proj.bias",
    "blk.0.attn_output.weight": "layers.0.self_attn.o_proj.weight",
    "blk.0.ffn_norm.weight": "layers.0.post_attention_layernorm.weight",
    "blk.0.ffn_gate.weight": "layers.0.mlp.gate_proj.weight",
    "blk.0.ffn_up.weight": "layers.0.mlp.up_proj.weight",
    "blk.0.ffn_down.weight": "layers.0.mlp.down_proj.weight",
}


def load_gguf_weights(gguf_path):
    reader = GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}
    out = {}
    for gguf_name, hf_name in WEIGHT_MAP.items():
        t = by_name[gguf_name]
        if t.tensor_type.name == "F32":
            arr = np.asarray(t.data, dtype=np.float32).reshape(-1)
        else:
            arr = dequantize(t.data, t.tensor_type).reshape(-1).astype(np.float32)
        out[hf_name] = arr
    return out


def build_patched_model(config_dir, weights):
    config = AutoConfig.from_pretrained(config_dir)
    config.num_hidden_layers = 1  # only block 0 is compared
    config._attn_implementation = "eager"  # sdpa does not expose attn probs
    torch.set_default_dtype(torch.float32)
    model = AutoModelForCausalLM.from_config(config)
    # config says torch_dtype bfloat16 and from_config may honor it; force
    # f32 BEFORE patching or the patched weights get rounded to bf16
    model = model.float()
    model.eval()
    dtype = next(model.parameters()).dtype
    assert dtype == torch.float32, f"model is {dtype}, references would be wrong"

    patched = set()
    with torch.no_grad():
        for hf_name, arr in weights.items():
            param = model.model.get_parameter(hf_name)
            param.copy_(torch.from_numpy(arr.reshape(param.shape)))
            patched.add(hf_name)
    assert patched == set(weights), "some weights were not patched"
    return model


def main():
    gguf_path, exe = sys.argv[1], sys.argv[2]
    config_dir = sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else "check_m5.ften"

    model = build_patched_model(config_dir, load_gguf_weights(gguf_path))
    seq = len(TOKENS)

    capt = {}

    def save(name):
        def hook(module, inputs, output):
            capt[name] = output.detach().clone()
        return hook

    def save_input(name):
        def hook(module, inputs):
            capt[name] = inputs[0].detach().clone()
        return hook

    layer = model.model.layers[0]
    model.model.embed_tokens.register_forward_hook(save("embeddings"))
    layer.input_layernorm.register_forward_hook(save("post_norm"))
    layer.self_attn.q_proj.register_forward_hook(save("q_proj"))
    layer.self_attn.k_proj.register_forward_hook(save("k_proj"))
    layer.self_attn.v_proj.register_forward_hook(save("v_proj"))
    layer.self_attn.o_proj.register_forward_pre_hook(save_input("attn_pre_o"))
    layer.self_attn.o_proj.register_forward_hook(save("attn_out"))
    layer.post_attention_layernorm.register_forward_hook(save("post_attn_norm"))
    layer.mlp.gate_proj.register_forward_hook(save("ffn_gate"))
    layer.mlp.up_proj.register_forward_hook(save("ffn_up"))
    layer.mlp.down_proj.register_forward_hook(save("ffn_out"))
    layer.register_forward_hook(
        lambda m, i, o: capt.__setitem__("block_out", o[0].detach().clone())
    )

    # RoPE is applied by a function, not a module, so wrap it; layer 0's call
    # is the first (and with 1 layer, the only) one
    original_rope = qwen_mod.apply_rotary_pos_emb

    def rope_wrapper(q, k, cos, sin, *args, **kwargs):
        q_rot, k_rot = original_rope(q, k, cos, sin, *args, **kwargs)
        capt.setdefault("q_rope", q_rot.detach().clone())
        capt.setdefault("k_rope", k_rot.detach().clone())
        return q_rot, k_rot

    qwen_mod.apply_rotary_pos_emb = rope_wrapper

    with torch.no_grad():
        result = model(torch.tensor([TOKENS]), output_attentions=True)
    attn_probs = result.attentions[0][0]  # [14, seq, seq], post-softmax

    def heads_to_flat(t):
        # [1, heads, seq, head_dim] -> [seq, heads*head_dim], matching our
        # row layout where head h occupies columns h*64..(h+1)*64
        return t.squeeze(0).transpose(0, 1).reshape(seq, -1)

    flat = {k: capt[k].squeeze(0) for k in (
        "embeddings", "post_norm", "q_proj", "k_proj", "v_proj",
        "attn_pre_o", "attn_out", "post_attn_norm",
        "ffn_gate", "ffn_up", "ffn_out", "block_out",
    )}
    tensors = {
        "tokens": torch.tensor(TOKENS, dtype=torch.float32),
        **flat,
        "q_rope": heads_to_flat(capt["q_rope"]),
        "k_rope": heads_to_flat(capt["k_rope"]),
        "attn_probs": attn_probs,
        "resid1": flat["embeddings"] + flat["attn_out"],
    }
    write_tensors(out_path, tensors)
    print(f"wrote {len(tensors)} reference tensors to {out_path}")

    if exe != "-":
        import subprocess
        sys.exit(subprocess.run([exe, "selftest-m5", gguf_path, out_path]).returncode)


if __name__ == "__main__":
    main()
