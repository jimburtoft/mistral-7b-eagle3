# SPDX-License-Identifier: Apache-2.0
"""CPU parity: functional target (eagle3/target.py) vs HF MistralForCausalLM, tiny random model.

Checks aux hidden states (HF hidden_states[i]) and final logits, for a right-padded batch and
for two documents packed into one row (each must equal the document run alone).
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eagle3.target import Target  # noqa: E402


def main():
    torch.manual_seed(0)
    from transformers import MistralConfig, MistralForCausalLM
    cfg = MistralConfig(hidden_size=64, intermediate_size=160, num_hidden_layers=6, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, vocab_size=257, rms_norm_eps=1e-5, rope_theta=1e6,
                        max_position_embeddings=512, sliding_window=None, tie_word_embeddings=False)
    hf = MistralForCausalLM(cfg).float().eval()
    for n, p in hf.named_parameters():
        if "norm" in n:
            p.data = 1 + 0.1 * torch.randn_like(p)
    aux_ids = [1, 3, 4]
    with tempfile.TemporaryDirectory() as d:
        hf.save_pretrained(d, safe_serialization=True)
        tg = Target(d, aux_ids, torch.device("cpu"), dtype=torch.float32)
    S = 40
    ids = torch.randint(0, 257, (1, S))
    with torch.no_grad():
        ref = hf(ids, output_hidden_states=True)
    doc = torch.zeros(1, S, dtype=torch.long)
    pos = torch.arange(S).unsqueeze(0)
    aux, final = tg.features(ids, doc, pos)
    ref_aux = torch.cat([ref.hidden_states[i] for i in aux_ids], -1)
    logits = final @ tg.lm_head.T
    e1 = (aux - ref_aux).abs().max().item()
    e2 = (logits - ref.logits).abs().max().item()
    print(f"single doc: aux max err {e1:.2e}, logits max err {e2:.2e}")
    assert e1 < 1e-4 and e2 < 1e-4
    # packed: doc A (25) + doc B (15) in one row == run separately
    a, b = torch.randint(0, 257, (1, 25)), torch.randint(0, 257, (1, 15))
    row = torch.cat([a, b], 1)
    doc = torch.cat([torch.zeros(25), torch.ones(15)]).long().unsqueeze(0)
    pos = torch.cat([torch.arange(25), torch.arange(15)]).unsqueeze(0)
    aux_p, fin_p = tg.features(row, doc, pos)
    with torch.no_grad():
        ra, rb = hf(a, output_hidden_states=True), hf(b, output_hidden_states=True)
    ref_cat = torch.cat([torch.cat([ra.hidden_states[i] for i in aux_ids], -1),
                         torch.cat([rb.hidden_states[i] for i in aux_ids], -1)], 1)
    e3 = (aux_p - ref_cat).abs().max().item()
    e4 = ((fin_p @ tg.lm_head.T) - torch.cat([ra.logits, rb.logits], 1)).abs().max().item()
    print(f"packed 2 docs: aux max err {e3:.2e}, logits max err {e4:.2e}")
    assert e3 < 1e-4 and e4 < 1e-4
    print("TARGET PARITY OK")


if __name__ == "__main__":
    main()
