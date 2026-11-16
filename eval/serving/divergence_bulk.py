# SPDX-License-Identifier: Apache-2.0
"""Greedy-identity audit over many outputs: for each base-vs-spec divergence, the base run's
top1-top2 logprob gap at the first diverging token (via the running BASE server).

Usage: python eval/serving/divergence_bulk.py --a base.json --b spec.json --prompts gsm8k --tokenizer <dir>
`--prompts gsm8k` rebuilds the GSM8K prompts used by gsm8k.py.
Prints a histogram of gaps; a functional error shows up as large gaps.
"""
import argparse
import concurrent.futures as cf
import json
import urllib.request

from gsm8k import SUF


def post(base, body):
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    qs = [r["question"] + SUF for r in load_dataset("openai/gsm8k", "main", split="test")]
    A, B = json.load(open(a.a)), json.load(open(a.b))
    jobs = []
    for i, (x, y) in enumerate(zip(A, B)):
        if x == y:
            continue
        ta = tok.encode(x, add_special_tokens=False)
        tb = tok.encode(y, add_special_tokens=False)
        n = next((k for k, (p, q) in enumerate(zip(ta, tb)) if p != q), min(len(ta), len(tb)))
        ids = tok.apply_chat_template([{"role": "user", "content": qs[i]}], tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        jobs.append((i, list(ids) + ta[:n]))

    def gap(job):
        r = post(a.base, {"model": "m", "prompt": job[1], "max_tokens": 1, "temperature": 0, "logprobs": 2})
        lp = sorted(r["choices"][0]["logprobs"]["top_logprobs"][0].values(), reverse=True)
        return job[0], (lp[0] - lp[1]) if len(lp) > 1 else 99.0

    with cf.ThreadPoolExecutor(16) as ex:
        gaps = list(ex.map(gap, jobs))
    vals = sorted(g for _, g in gaps)
    bins = [0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 99]
    hist = {f"<={b}": sum(1 for v in vals if v <= b) for b in bins}
    res = {"n": len(A), "identical": len(A) - len(jobs), "diverged": len(jobs), "gap_cumulative": hist,
           "max_gap": vals[-1] if vals else None, "worst": sorted(gaps, key=lambda x: -x[1])[:5]}
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
