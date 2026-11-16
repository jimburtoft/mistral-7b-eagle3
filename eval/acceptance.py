# SPDX-License-Identifier: Apache-2.0
"""Offline EAGLE3 acceptance on held-out conversations (teacher-forced on the target's greedy output).

For each held-out conversation (prompt + target-regenerated greedy response), at every response
position t the draft proposes k tokens autoregressively, exactly as at serving time:
  step 0: fc(aux[t]) with token x[t+1]                      -> d1
  step j: draft's own hidden from step j-1 with token d_j    -> d_{j+1}
Acceptance length at t = 1 + number of leading proposals equal to the target's greedy tokens
x[t+2..t+k+1]. Because the reference response IS the target's greedy output, this equals the
serving acceptance length under greedy verification (up to bf16 tie-breaking).

Runs every position in parallel: step j attends to the context keys (causal, from step 0) plus the
draft's own keys from steps 1..j at the same position (the TTT "diagonal" structure) -- the same
computation as training, but with the draft's predicted tokens instead of teacher forcing.

Output: mean acceptance length (k=1..K), per-depth conditional acceptance, per-source breakdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eagle3.batch import gather_rows, shifted_cpu  # noqa: E402
from eagle3.checkpoint import load_speculators  # noqa: E402
from eagle3.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from eagle3.model import build_mask, rope_cos_sin  # noqa: E402
from eagle3.target import Target  # noqa: E402
from train.producer import load_rows  # noqa: E402


class Propose(torch.nn.Module):
    def __init__(self, draft, k):
        super().__init__()
        self.d, self.k = draft, k

    def forward(self, aux, ids, doc, pos):
        d = self.d
        mask0 = build_mask(doc)
        h = d.fc(aux)
        kv: list = []
        cur = ids
        props = []
        for j in range(self.k):
            cos, sin = rope_cos_sin(pos + j, d.c.head_dim, d.c.rope_theta, h.dtype)
            h, logits = d.step(cur, h, cos, sin, mask0, kv)
            tok = logits.argmax(-1)
            tok = tok + d.d2t[tok] if d.c.draft_vocab_size != d.c.vocab_size else tok
            props.append(tok)
            cur = tok
        return torch.stack(props, -1)  # [B, S, k] target-vocab ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--rows", required=True, help="packed held-out rows (data/pack.py on regen_test.jsonl)")
    ap.add_argument("--sources", default="", help="optional JSON list: source label per document, in row order")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-compile", action="store_true")
    add_device_arg(ap)
    a = ap.parse_args()
    dev = get_device(a.device)
    draft = load_speculators(a.draft)
    aux_ids = draft.c.aux_layer_ids
    draft = draft.to(dev, torch.bfloat16).eval()
    tg = Target(a.target, aux_ids, dev, compile_fn=None if a.no_compile else (lambda f: maybe_compile(f, dev)))
    prop = maybe_compile(Propose(draft, a.k), dev, enabled=not a.no_compile)
    rows = load_rows(a.rows)
    K = a.k
    tot = torch.zeros(K + 1, dtype=torch.float64)  # tot[j] = # positions with >= j accepted
    n_pos = 0
    with torch.no_grad():
        for r in range(rows["ids"].shape[0]):
            sl = slice(r, r + 1)
            ids, doc, pos, lm, gi = (rows[x][sl] for x in ("ids", "doc", "pos", "lm", "gi"))
            ids, doc, pos = ids.long(), doc.long(), pos.long()
            ids_s, doc_s, pos_s, lm_s, g, _ = shifted_cpu(ids, doc, pos, lm, gi)
            aux, _ = tg.features(ids.to(dev), doc.to(dev), pos.to(dev))
            aux_s = gather_rows(aux.cpu(), g)
            p = prop(aux_s.to(dev), ids_s.to(dev), doc_s.to(dev), pos_s.to(dev)).cpu()  # [1,S,K]
            S = ids_s.shape[1]
            # reference: target greedy tokens x[t+2..t+K+1] = ids_s[t+1..t+K] (same document)
            ref = torch.full((1, S, K), -1, dtype=torch.long)
            ok_ref = torch.zeros(1, S, K, dtype=torch.bool)
            for j in range(K):
                if j + 1 < S:
                    ref[:, : S - 1 - j, j] = ids_s[:, 1 + j:]
                    ok_ref[:, : S - 1 - j, j] = (doc_s[:, 1 + j:] == doc_s[:, : S - 1 - j]) & lm_s[:, 1 + j:]
            hit = (p == ref) & ok_ref
            run = hit.long().cumprod(-1)                     # leading accepted
            # positions where the next token (t+1) is a response token and all K references exist
            valid = lm_s & ok_ref.all(-1)
            n_pos += int(valid.sum())
            tot[0] += int(valid.sum())
            for j in range(K):
                tot[j + 1] += int((run[..., j].bool() & valid).sum())
    res = {"positions": n_pos}
    for k in range(1, K + 1):
        res[f"accept_len_k{k}"] = round(float(1 + tot[1:k + 1].sum() / max(n_pos, 1)), 4)
    res["cond_acc"] = [round(float(tot[j + 1] / max(tot[j], 1)), 4) for j in range(K)]
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
