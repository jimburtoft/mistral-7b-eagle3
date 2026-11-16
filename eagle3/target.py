# SPDX-License-Identifier: Apache-2.0
"""Frozen target (verifier) feature extraction for EAGLE3 training. CPU / CUDA / Trainium.

A functional Llama/Mistral forward over the target's safetensors weights: one decoder-layer
function takes the layer weights as tensor arguments, so a single compiled graph serves all
layers (no per-layer recompiles). Matches HF ``MistralForCausalLM`` (checked by
tests/parity_target.py) for checkpoints without a sliding window.

Outputs per token (right-padded rows, or packed rows with a block-diagonal mask):
  aux   [B, S, n_aux*H]  hidden state *entering* each layer in ``aux_layer_ids``
                         (HF ``hidden_states[i]``; the vLLM / vllm-neuron EAGLE3 convention)
  final [B, S, H]        final-norm output (target logits = final @ lm_head.T)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import apply_rope, rope_cos_sin


@dataclass
class TargetConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int

    @classmethod
    def from_hf(cls, path):
        c = json.load(open(os.path.join(path, "config.json")))
        assert c.get("sliding_window") in (None, 0) or c["sliding_window"] >= c["max_position_embeddings"], \
            "sliding-window targets are not supported"
        rs = c.get("rope_scaling") or c.get("rope_parameters") or {}
        assert rs.get("rope_type", rs.get("type", "default")) in ("default", None), f"unsupported rope {rs}"
        H, nh = c["hidden_size"], c["num_attention_heads"]
        return cls(H, c["intermediate_size"], c["num_hidden_layers"], nh, c["num_key_value_heads"],
                   c.get("head_dim") or H // nh, c["rms_norm_eps"],
                   c.get("rope_theta", rs.get("rope_theta", 10000.0)), c["vocab_size"])


def rms(x, w, eps):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return w * x.to(dt)


LAYER_KEYS = ("input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
              "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
              "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def decoder_layer(h, cos, sin, mask, n1, wq, wk, wv, wo, n2, wg, wu, wd, *, nh: int, nkv: int, hd: int, eps: float):
    B, S, _ = h.shape
    x = rms(h, n1, eps)
    q = F.linear(x, wq).view(B, S, nh, hd).transpose(1, 2)
    k = F.linear(x, wk).view(B, S, nkv, hd).transpose(1, 2)
    v = F.linear(x, wv).view(B, S, nkv, hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    rep = nh // nkv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    s = torch.matmul(q, k.transpose(-1, -2)).float() * (hd ** -0.5) + mask
    o = torch.matmul(torch.softmax(s, dim=-1).to(q.dtype), v).transpose(1, 2).reshape(B, S, nh * hd)
    h = h + F.linear(o, wo)
    x = rms(h, n2, eps)
    return h + F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)


class Target:
    """Holds target weights on `device`; `features(ids, doc, pos)` -> (aux, final)."""

    def __init__(self, path: str, aux_layer_ids, device, dtype=torch.bfloat16, compile_fn=None,
                 max_layers: int | None = None):
        from safetensors import safe_open
        self.c = TargetConfig.from_hf(path)
        self.aux = list(aux_layer_ids)
        self.n_layers = max_layers or self.c.num_hidden_layers
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"] \
            if os.path.exists(os.path.join(path, "model.safetensors.index.json")) else None
        files = sorted(set(idx.values())) if idx else ["model.safetensors"]
        w = {}
        for f in files:
            with safe_open(os.path.join(path, f), "pt") as fh:
                for k in fh.keys():
                    if k.startswith("model.layers."):
                        li = int(k.split(".")[2])
                        if li >= self.n_layers:
                            continue
                    w[k] = fh.get_tensor(k)
        self.embed = w["model.embed_tokens.weight"].to(device, dtype)
        self.norm = w["model.norm.weight"].to(device, dtype)
        self.lm_head = w.get("lm_head.weight", w["model.embed_tokens.weight"]).to(dtype)  # kept on CPU
        self.layers = [[w[f"model.layers.{i}.{k}"].to(device, dtype) for k in LAYER_KEYS] for i in range(self.n_layers)]
        del w
        self.device, self.dtype = device, dtype
        c = self.c
        fn = lambda *a: decoder_layer(*a, nh=c.num_attention_heads, nkv=c.num_key_value_heads, hd=c.head_dim,
                                      eps=c.rms_norm_eps)
        self.layer_fn = compile_fn(fn) if compile_fn else fn

    @torch.no_grad()
    def features(self, ids, doc, pos):
        """ids/doc/pos [B,S] on device (doc -1 = pad, pos = position within document)."""
        from .model import build_mask
        h = F.embedding(ids, self.embed)
        cos, sin = rope_cos_sin(pos, self.c.head_dim, self.c.rope_theta, self.dtype)
        mask = build_mask(doc)
        aux = []
        for li, lw in enumerate(self.layers):
            if li in self.aux:
                aux.append(h)
            h = self.layer_fn(h, cos, sin, mask, *lw)
        return torch.cat(aux, dim=-1), rms(h, self.norm, self.c.rms_norm_eps)
