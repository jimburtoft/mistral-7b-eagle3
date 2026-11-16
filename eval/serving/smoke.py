# SPDX-License-Identifier: Apache-2.0
"""Greedy identity smoke test against an OpenAI-compatible server.

Runs N fixed prompts at temperature 0 and writes {prompt_id: text} JSON.
With --compare, diffs against a previous run (e.g. the no-speculation baseline) and
reports exact-match count and first divergence position.
"""
import argparse
import concurrent.futures as cf
import json
import urllib.request

PROMPTS = [
    "I am gonna keep counting forever, 1 2 3 4 5 ",
    "Explain the difference between a process and a thread in three sentences.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What is the capital of Australia, and why was it chosen?",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "List five common sorting algorithms and their average time complexity.",
    "Translate 'The weather is nice today' into French, Spanish and German.",
    "A train leaves at 3pm travelling 60 km/h. Another leaves the same station at 4pm at 90 km/h. When does the second catch up?",
]


def chat(base, prompt, max_tokens):
    body = json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        outs = list(ex.map(lambda p: chat(a.base, p, a.max_tokens), PROMPTS))
    res = {str(i): t for i, t in enumerate(outs)}
    json.dump(res, open(a.out, "w"), indent=1)
    print("first:", repr(outs[0][:80]))
    if a.compare:
        ref = json.load(open(a.compare))
        same = 0
        for k, t in res.items():
            r = ref[k]
            if t == r:
                same += 1
            else:
                d = next((i for i, (x, y) in enumerate(zip(t, r)) if x != y), min(len(t), len(r)))
                print(f"prompt {k}: diverges at char {d}: {t[d:d+40]!r} vs {r[d:d+40]!r}")
        print(f"identical {same}/{len(res)}")


if __name__ == "__main__":
    main()
