# SPDX-License-Identifier: Apache-2.0
"""GSM8K (test, 1319) through an OpenAI-compatible server: greedy, zero-shot CoT, chat endpoint.

Usage: python eval/serving/gsm8k.py --tag <tag> [--n N] [--concurrency C] [--out-dir results]
Writes <out-dir>/gsm8k-<tag>.json (outputs) and prints accuracy.
"""
import argparse
import concurrent.futures as cf
import json
import re
import urllib.request

SUF = "\nSolve the problem step by step. End your response with 'The answer is N.' where N is a number."


def ask(base, q, max_tokens):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": q + SUF}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"]["content"] or ""


def pred(t):
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", t) or re.findall(r"-?[\d,]*\.?\d+", t)
    return m[-1].replace(",", "").rstrip(".") if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = list(ds)[: a.n or None]
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        outs = list(ex.map(lambda q: ask(a.base, q, a.max_tokens), [i["question"] for i in items]))
    ok = 0
    for it, t in zip(items, outs):
        g = it["answer"].split("####")[-1].strip().replace(",", "")
        p = pred(t)
        try:
            ok += p is not None and abs(float(p) - float(g)) < 1e-6
        except ValueError:
            pass
    json.dump(outs, open(f"{a.out_dir}/gsm8k-{a.tag}.json", "w"))
    print(f"GSM8K {a.tag} n={len(items)} acc={ok / len(items):.4f}")


if __name__ == "__main__":
    main()
