# SPDX-License-Identifier: Apache-2.0
"""Tokenize regenerated conversations and pack them into fixed-length rows.

Each conversation = [user prompt, assistant response] rendered with the target's chat
template. loss_mask covers the assistant response tokens (+ EOS). Documents are packed
first-fit-decreasing into rows of --seq-len tokens; a document longer than seq-len is
truncated. Per row we store (int32 / uint8, numpy .npz shards):

  ids [S]   doc [S] (-1 = pad)   pos [S] (position within its document)   lm [S] (loss mask)
  gi  [S]   shifted-layout gather index (see eagle3/batch.py), -1 = pad

Device-agnostic (CPU only, tokenizer from transformers).
"""

import argparse
import json
import os

import numpy as np


def render(tok, prompt, response):
    """Token ids for the full conversation and the length of the prompt part."""
    p = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                                   tokenize=True)
    if hasattr(p, "keys"):
        p, full = p["input_ids"], full["input_ids"]
    p, full = list(p), list(full)
    assert full[:len(p)] == p, "chat template: prompt is not a prefix of the full conversation"
    return full, len(p)


def shift_index(doc: np.ndarray) -> np.ndarray:
    """gi[k] = source position of the k-th shifted slot (pairs aux[gi] with token/target gi+1).

    Per document of length L the shifted layout has L-1 slots (speculators shift_batch applied
    per document). Slots are laid out in document order; the rest is -1.
    """
    S = doc.shape[0]
    gi = np.full(S, -1, dtype=np.int32)
    k = 0
    for t in range(S - 1):
        if doc[t] >= 0 and doc[t + 1] == doc[t]:
            gi[k] = t
            k += 1
    return gi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="regenerated JSONL (data/regenerate.py output)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, help="output dir for row shards")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--rows-per-shard", type=int, default=2048)
    ap.add_argument("--drop-truncated", action="store_true", help="drop responses that hit max_tokens")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    S = a.seq_len
    docs = []
    n_trunc = n_drop = 0
    for line in open(a.inp):
        r = json.loads(line)
        if a.drop_truncated and r.get("finish_reason") == "length":
            n_drop += 1
            continue
        ids, plen = render(tok, r["prompt"], r["response"])
        if plen >= S - 8:
            n_drop += 1
            continue
        if len(ids) > S:
            ids = ids[:S]
            n_trunc += 1
        lm = np.zeros(len(ids), dtype=np.uint8)
        lm[plen:] = 1
        docs.append((np.asarray(ids, dtype=np.int32), lm))
    # first-fit decreasing packing
    order = sorted(range(len(docs)), key=lambda i: -len(docs[i][0]))
    rows, space = [], []
    for i in order:
        L = len(docs[i][0])
        for r, free in enumerate(space):
            if free >= L:
                rows[r].append(i)
                space[r] -= L
                break
        else:
            rows.append([i])
            space.append(S - L)
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    os.makedirs(a.out, exist_ok=True)
    n_tok = n_lm = 0
    for s0 in range(0, len(rows), a.rows_per_shard):
        chunk = rows[s0:s0 + a.rows_per_shard]
        R = len(chunk)
        ids = np.zeros((R, S), np.int32)
        doc = np.full((R, S), -1, np.int32)
        pos = np.zeros((R, S), np.int32)
        lm = np.zeros((R, S), np.uint8)
        gi = np.full((R, S), -1, np.int32)
        for r, members in enumerate(chunk):
            t = 0
            for d, i in enumerate(members):
                x, m = docs[i]
                L = len(x)
                ids[r, t:t + L] = x
                doc[r, t:t + L] = d
                pos[r, t:t + L] = np.arange(L)
                lm[r, t:t + L] = m
                t += L
            gi[r] = shift_index(doc[r])
            n_tok += t
            n_lm += int(lm[r].sum())
        np.savez(os.path.join(a.out, f"rows_{s0 // a.rows_per_shard:04d}.npz"), ids=ids, doc=doc, pos=pos, lm=lm, gi=gi)
    stats = {"docs": len(docs), "rows": len(rows), "seq_len": S, "tokens": n_tok, "loss_tokens": n_lm,
             "fill": round(n_tok / max(1, len(rows) * S), 4), "truncated": n_trunc, "dropped": n_drop}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
