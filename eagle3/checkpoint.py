# SPDX-License-Identifier: Apache-2.0
"""Save / load the draft in the speculators EAGLE3 checkpoint format.

The output directory (config.json + model.safetensors) loads in vLLM and vllm-neuron with
``--speculative-config '{"method": "eagle3", "model": <dir>, "num_speculative_tokens": k}'``.
Format reference: speculators v0.9.0 (Apache-2.0) and the public
RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3 checkpoint.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file, save_file

from .model import Eagle3Config, Eagle3Draft


def save_speculators(draft: Eagle3Draft, out_dir: str, verifier_name: str, verifier_config: dict,
                     ttt_steps: int = 3, dtype=torch.bfloat16):
    os.makedirs(out_dir, exist_ok=True)
    c = draft.c
    sd = {}
    for k, v in draft.state_dict().items():
        sd[k] = v.detach().to("cpu", dtype).contiguous()
    sd["embed_tokens.weight"] = draft.embed_tokens.detach().to("cpu", dtype).contiguous()
    sd["lm_head.weight"] = draft.lm_head.detach().to("cpu", dtype).contiguous()
    if c.draft_vocab_size != c.vocab_size:
        sd["d2t"] = draft.d2t.cpu().long()
        sd["t2d"] = draft.t2d.cpu().bool()
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    tl = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": c.hidden_size, "intermediate_size": c.intermediate_size,
        "num_attention_heads": c.num_attention_heads, "num_key_value_heads": c.num_key_value_heads,
        "head_dim": c.head_dim, "num_hidden_layers": 1, "rms_norm_eps": c.rms_norm_eps,
        "rope_theta": c.rope_theta, "rope_scaling": None, "max_position_embeddings": c.max_position_embeddings,
        "vocab_size": c.vocab_size, "hidden_act": "silu", "attention_bias": False, "mlp_bias": False,
        "attention_dropout": 0.0, "initializer_range": 0.02, "pretraining_tp": 1, "tie_word_embeddings": False,
        "torch_dtype": "bfloat16", "bos_token_id": verifier_config.get("bos_token_id"),
        "eos_token_id": verifier_config.get("eos_token_id"),
    }
    cfg = {
        "architectures": ["Eagle3Speculator"],
        "speculators_model_type": "eagle3",
        "speculators_version": "0.9.0",
        "draft_vocab_size": c.draft_vocab_size,
        "norm_before_residual": c.norm_before_residual,
        "norm_before_fc": False, "fc_norm": False, "norm_output": False, "embed_requires_grad": False,
        "target_hidden_size": None,
        "eagle_aux_hidden_state_layer_ids": list(c.aux_layer_ids),
        "has_no_defaults_at_init": False,
        "torch_dtype": "bfloat16",
        "transformer_layer_config": tl,
        "speculators_config": {
            "algorithm": "eagle3",
            "default_proposal_method": "greedy",
            "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": ttt_steps,
                                  "verifier_accept_k": 1, "accept_tolerance": 0.0}],
            "verifier": {"name_or_path": verifier_name,
                         "architectures": verifier_config.get("architectures", ["MistralForCausalLM"])},
        },
    }
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)


def load_speculators(path: str) -> Eagle3Draft:
    cfg = json.load(open(os.path.join(path, "config.json")))
    tl = cfg["transformer_layer_config"]
    c = Eagle3Config(hidden_size=tl["hidden_size"], intermediate_size=tl["intermediate_size"],
                     num_attention_heads=tl["num_attention_heads"], num_key_value_heads=tl["num_key_value_heads"],
                     head_dim=tl.get("head_dim") or tl["hidden_size"] // tl["num_attention_heads"],
                     rms_norm_eps=tl["rms_norm_eps"], rope_theta=tl["rope_theta"],
                     max_position_embeddings=tl["max_position_embeddings"], vocab_size=tl["vocab_size"],
                     draft_vocab_size=cfg["draft_vocab_size"], aux_layer_ids=cfg["eagle_aux_hidden_state_layer_ids"],
                     norm_before_residual=cfg["norm_before_residual"])
    d = Eagle3Draft(c)
    sd = load_file(os.path.join(path, "model.safetensors"))
    emb, lm = sd.pop("embed_tokens.weight"), sd.pop("lm_head.weight")
    d2t, t2d = sd.pop("d2t", None), sd.pop("t2d", None)
    d.load_state_dict({k: v.float() for k, v in sd.items()})
    if t2d is None:
        t2d = torch.ones(c.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(c.draft_vocab_size, dtype=torch.long)
        full = lm
    else:
        full = torch.zeros(c.vocab_size, c.hidden_size, dtype=lm.dtype)
        full[t2d] = lm
    d.set_frozen(emb, full, d2t, t2d)
    return d
