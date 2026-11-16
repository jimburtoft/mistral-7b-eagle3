# SPDX-License-Identifier: Apache-2.0
"""Logprob margin at the first divergence between two greedy runs.

For each diverging prompt, asks the (running) server for the next-token
top-5 logprobs after the common prefix. A small top1-top2 gap means the
divergence is a numerical near-tie, not a functional error.
"""
import argparse
import json
import urllib.request

from smoke import PROMPTS


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tokenizer", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    A, B = json.load(open(a.a)), json.load(open(a.b))
    for k in A:
        if A[k] == B[k]:
            continue
        ta = tok.encode(A[k], add_special_tokens=False)
        tb = tok.encode(B[k], add_special_tokens=False)
        n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        ids = tok.apply_chat_template([{"role": "user", "content": PROMPTS[int(k)]}], tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        prompt = list(ids) + ta[:n]
        r = post(a.base, "/v1/completions", {"model": "m", "prompt": prompt, "max_tokens": 1,
                                             "temperature": 0, "logprobs": 5})
        lp = r["choices"][0]["logprobs"]["top_logprobs"][0]
        top = sorted(lp.items(), key=lambda x: -x[1])[:3]
        gap = top[0][1] - top[1][1] if len(top) > 1 else float("inf")
        print(f"prompt {k} tok {n}: A={tok.decode([ta[n]])!r} B={tok.decode([tb[n]])!r} "
              f"top3={[(t, round(v, 3)) for t, v in top]} gap={gap:.3f}")


if __name__ == "__main__":
    main()
