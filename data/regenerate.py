# SPDX-License-Identifier: Apache-2.0
"""Regenerate assistant responses with the target model (greedy).

Talks to any OpenAI-compatible server (vllm-neuron on Trainium, or vLLM on GPU),
so the script is device-agnostic. Resumable: rows already in --out are skipped.

Output JSONL rows: {"id", "source", "prompt", "response", "finish_reason", "n_out"}
"""
import argparse
import concurrent.futures as cf
import json
import os
import threading
import time
import urllib.request


def chat(base, model, prompt, max_tokens, timeout=1800):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                j = json.load(r)
            c = j["choices"][0]
            return c["message"]["content"], c["finish_reason"], j["usage"]["completion_tokens"]
        except urllib.error.HTTPError as e:
            if e.code == 400:  # prompt too long for the server, etc.: skip
                return None, f"http400:{e.read()[:200]!r}", 0
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None, "failed", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="m")
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    rows = [json.loads(l) for l in open(a.inp)]
    if a.limit:
        rows = rows[:a.limit]
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} prompts, {len(done)} done, {len(todo)} to go", flush=True)

    lock = threading.Lock()
    t0, n_tok, n = time.time(), 0, 0
    with open(a.out, "a") as fo, cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(chat, a.base, a.model, r["prompt"], a.max_tokens): r for r in todo}
        for fut in cf.as_completed(futs):
            r = futs[fut]
            text, fin, k = fut.result()
            if text is None:
                print("skip", r["id"], fin, flush=True)
                continue
            with lock:
                fo.write(json.dumps({**r, "response": text, "finish_reason": fin, "n_out": k},
                                    ensure_ascii=False) + "\n")
                n += 1
                n_tok += k
                if n % 500 == 0:
                    fo.flush()
                    dt = time.time() - t0
                    print(f"{n}/{len(todo)}  {n_tok / dt:.0f} out tok/s  eta {(len(todo) - n) * dt / n / 3600:.2f} h",
                          flush=True)
    print(f"done {n} rows, {n_tok} tokens, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
