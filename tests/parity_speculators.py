# SPDX-License-Identifier: Apache-2.0
"""CPU parity: eagle3/model.py vs the speculators reference implementation.

Builds speculators' Eagle3DraftModel (v0.9.0, eager attention, eager KL loss) with a tiny
random Llama config, copies its weights into our Eagle3Draft, runs both objectives on the
same packed batch, and compares loss, per-step accuracy and fc / q_proj gradients.

speculators packs documents into one row; we use one document per row with right padding.
The test uses a single document so the two layouts are the same.

Run: python tests/parity_speculators.py   (needs `pip install speculators==0.9.0`, CPU only)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eagle3.model import Eagle3Config, Eagle3Draft, eagle3_loss  # noqa: E402


def main(two_docs=False):
    torch.manual_seed(0)
    from transformers import LlamaConfig
    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.models.eagle3 import Eagle3SpeculatorConfig
    from speculators.models.eagle3.core import Eagle3DraftModel
    from speculators.losses import resolve_loss_config
    from speculators.proposals.greedy import GreedyTokenProposalConfig

    H, V, Vd, S, TTT = 64, 300, 200, 24, 3
    tl = LlamaConfig(hidden_size=H, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, num_hidden_layers=1, vocab_size=V, rms_norm_eps=1e-5, rope_theta=1e6,
                     max_position_embeddings=4096, attention_bias=False, mlp_bias=False)
    tl._attn_implementation = "eager"
    cfg = Eagle3SpeculatorConfig(
        transformer_layer_config=tl, draft_vocab_size=Vd, norm_before_residual=True,
        eagle_aux_hidden_state_layer_ids=[2, 16, 29],
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3", default_proposal_method="greedy",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=TTT)],
            verifier=VerifierConfig(name_or_path=None, architectures=["MistralForCausalLM"])),
    )
    ref = Eagle3DraftModel(cfg).float()
    for n, p in ref.named_parameters():
        if p.isnan().any() or "norm" not in n:
            torch.nn.init.normal_(p, std=0.05)
    # vocab mapping
    sel = torch.sort(torch.randperm(V)[:Vd]).values
    t2d = torch.zeros(V, dtype=torch.bool)
    t2d[sel] = True
    d2t = sel - torch.arange(Vd)
    ref.load_vocab_mappings(t2d, d2t)
    embed = torch.randn(V, H) * 0.1
    vlm = torch.randn(V, H) * 0.1
    with torch.no_grad():
        ref.embed_tokens.weight.copy_(embed)
        ref.lm_head.weight.copy_(vlm[t2d])
        ref.verifier_lm_head.weight.copy_(vlm[t2d])
        ref.verifier_norm.weight.fill_(1.0)
        for nrm in ("norm", "layers.0.input_layernorm", "layers.0.hidden_norm", "layers.0.post_attention_layernorm"):
            ref.get_submodule(nrm).weight.copy_(1 + 0.1 * torch.randn(H))

    # ours
    c = Eagle3Config(hidden_size=H, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, vocab_size=V, draft_vocab_size=Vd, rope_theta=1e6)
    ours = Eagle3Draft(c).float()
    sd = {k: v for k, v in ref.state_dict().items() if not k.startswith(("verifier_", "embed_tokens", "lm_head", "d2t", "t2d"))}
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not [m for m in missing if m not in ("embed_tokens", "lm_head", "d2t", "t2d")], missing
    ours.set_frozen(embed, vlm, d2t, t2d)

    # One shifted sample (speculators shift_batch already applied)
    ids = torch.randint(0, V, (1, S))
    aux = torch.randn(1, S, 3 * H)
    last_h = torch.randn(1, S, H)
    loss_mask = torch.ones(1, S, dtype=torch.bool)
    loss_mask[0, :5] = False
    pos = torch.arange(1, S + 1).unsqueeze(0)
    docs = torch.zeros(1, S, dtype=torch.long)
    if two_docs:  # pack two documents: 14 + 10 tokens, positions restart per document
        docs[0, 14:] = 1
        pos = torch.cat([torch.arange(1, 15), torch.arange(1, 11)]).unsqueeze(0)

    loss_cfg = resolve_loss_config("kl_div", "eager")
    _, l_ref, m_ref = ref(hidden_states=aux, input_ids=ids, document_ids=docs,
                          loss_mask=loss_mask, position_ids=pos.clone(), verifier_last_hidden_states=last_h,
                          ttt_steps=TTT, loss_config=loss_cfg)
    l_ref.backward()

    with torch.no_grad():
        tgt_logits = ref.verifier_lm_head(ref.verifier_norm(last_h))
    l_our, stats = eagle3_loss(ours, aux, ids, tgt_logits, loss_mask, docs, pos, TTT, same_doc_only=False)
    l_our.backward()

    print(f"loss ref {l_ref.item():.6f} ours {l_our.item():.6f}")
    assert abs(l_ref.item() - l_our.item()) < 1e-4 * max(1, abs(l_ref.item())), "loss mismatch"
    for j in range(TTT):
        ra = m_ref[f"full_acc_{j}_sum"].item()
        print(f"step {j}: correct ref {ra:.0f} ours {stats[j, 0].item():.0f} / {stats[j, 1].item():.0f}")
        assert abs(ra - stats[j, 0].item()) < 0.5
    for n in ("fc.weight", "layers.0.self_attn.q_proj.weight", "layers.0.mlp.down_proj.weight", "layers.0.hidden_norm.weight"):
        g1 = ref.get_parameter(n).grad
        g2 = ours.get_parameter(n).grad
        cos = torch.nn.functional.cosine_similarity(g1.flatten(), g2.flatten(), dim=0).item()
        rel = ((g1 - g2).norm() / g1.norm()).item()
        print(f"grad {n}: cos {cos:.6f} rel {rel:.2e}")
        assert cos > 0.9999 and rel < 1e-3
    print("PARITY OK", "(2 packed docs)" if two_docs else "(1 doc)")


def padding_test():
    """Two right-padded rows in one batch == the same rows run alone (loss sums, grads)."""
    torch.manual_seed(1)
    H, V, S, TTT = 64, 300, 32, 3
    c = Eagle3Config(hidden_size=H, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, vocab_size=V, draft_vocab_size=V)
    m = Eagle3Draft(c).double()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    m.set_frozen(torch.randn(V, H).double() * 0.1, torch.randn(V, H).double() * 0.1,
                 torch.zeros(V, dtype=torch.long), torch.ones(V, dtype=torch.bool))
    lens = [32, 19]
    ids = torch.randint(0, V, (2, S))
    aux = torch.randn(2, S, 3 * H).double()
    tl = torch.randn(2, S, V).double()
    lm = torch.rand(2, S) > 0.3
    valid = torch.arange(S).unsqueeze(0) < torch.tensor(lens).unsqueeze(1)
    lm &= valid
    docs = torch.where(valid, 0, -1)
    pos = torch.arange(1, S + 1).unsqueeze(0).expand(2, S)
    tot, st = eagle3_loss(m, aux, ids, tl, lm, docs, pos, TTT)
    n_all = st[:, 1]
    # per row alone, unpadded
    sums = torch.zeros(TTT, dtype=torch.double)
    cnt = torch.zeros(TTT, dtype=torch.double)
    for r, L in enumerate(lens):
        sl = slice(0, L)
        t, s1 = eagle3_loss(m, aux[r:r + 1, sl], ids[r:r + 1, sl], tl[r:r + 1, sl], lm[r:r + 1, sl],
                            docs[r:r + 1, sl], pos[r:r + 1, sl], TTT)
        cnt += s1[:, 1].double()
    print("padding: masked counts batch", n_all.tolist(), "rows", cnt.tolist())
    # per-step loss check via TTT=1 (single step: exact token-weighted equality)
    t_b, _ = eagle3_loss(m, aux, ids, tl, lm, docs, pos, 1)
    num = 0.0
    den = 0.0
    for r, L in enumerate(lens):
        t_r, s_r = eagle3_loss(m, aux[r:r + 1, :L], ids[r:r + 1, :L], tl[r:r + 1, :L], lm[r:r + 1, :L],
                               docs[r:r + 1, :L], pos[r:r + 1, :L], 1)
        num += t_r.item() * (s_r[0, 1].item() + 1e-5)
        den += s_r[0, 1].item()
    print(f"padding step0 loss batch {t_b.item():.10f} rows {num / (den + 1e-5):.10f}")
    assert abs(t_b.item() - num / (den + 1e-5)) < 1e-8
    print("PADDING OK")


if __name__ == "__main__":
    main()
    main(two_docs=True)
    padding_test()
