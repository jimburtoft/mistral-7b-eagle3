# SPDX-License-Identifier: Apache-2.0
"""Build the prompt set for EAGLE3 draft training.

Sources (all openly licensed; see data/LICENSES.md):
  ultrachat  HuggingFaceH4/ultrachat_200k   (MIT)         general multi-domain chat
  magicoder  ise-uiuc/Magicoder-OSS-Instruct-75K (MIT)    code instructions
  gsm8k      openai/gsm8k (main)            (MIT)         grade-school math word problems

Only the first user turn is kept; responses are regenerated later by the target
model (data/regenerate.py), so no third-party assistant text is used for training.

Output: JSONL rows {"id", "source", "prompt"} for train and test splits.
The UltraChat test split is drawn from `test_sft` (held out by the dataset authors);
GSM8K test from `test`; Magicoder has no test split, so a fixed hashed slice is held out.
"""
import argparse
import hashlib
import json
import os
import random

from datasets import load_dataset

SOURCES = {
    "ultrachat": dict(repo="HuggingFaceH4/ultrachat_200k", config=None,
                      train="train_sft", test="test_sft"),
    "magicoder": dict(repo="ise-uiuc/Magicoder-OSS-Instruct-75K", config=None,
                      train="train", test=None),
    "gsm8k": dict(repo="openai/gsm8k", config="main", train="train", test="test"),
}


def first_prompt(source, row):
    if source == "ultrachat":
        return row["messages"][0]["content"]
    if source == "magicoder":
        return row["problem"]
    if source == "gsm8k":
        return row["question"]
    raise KeyError(source)


def norm_key(text):
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()


def is_held_out(key, frac):
    return int(key[:8], 16) / 0xFFFFFFFF < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/prompts")
    ap.add_argument("--n-train", default="ultrachat=60000,magicoder=20000,gsm8k=7473")
    ap.add_argument("--n-test", default="ultrachat=500,magicoder=200,gsm8k=300")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n_train = {k: int(v) for k, v in (x.split("=") for x in a.n_train.split(","))}
    n_test = {k: int(v) for k, v in (x.split("=") for x in a.n_test.split(","))}
    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    seen = set()
    rows = {"train": [], "test": []}
    manifest = {}

    # Test prompts first so that train never contains a test prompt (dedupe by normalized text).
    for split in ("test", "train"):
        for src, cfg in SOURCES.items():
            want = (n_test if split == "test" else n_train).get(src, 0)
            if want == 0:
                continue
            hf_split = cfg[split] or cfg["train"]
            ds = load_dataset(cfg["repo"], cfg["config"], split=hf_split)
            manifest.setdefault(src, {"repo": cfg["repo"], "config": cfg["config"],
                                      "fingerprint": {}})["fingerprint"][hf_split] = ds._fingerprint
            idx = list(range(len(ds)))
            rng.shuffle(idx)
            got = 0
            for i in idx:
                p = first_prompt(src, ds[i]).strip()
                if not (a.min_chars <= len(p) <= a.max_chars):
                    continue
                k = norm_key(p)
                if k in seen:
                    continue
                if cfg[split] is None:  # no native test split: deterministic hashed hold-out
                    if (split == "test") != is_held_out(k, 0.02):
                        continue
                seen.add(k)
                rows[split].append({"id": f"{src}-{hf_split}-{i}", "source": src, "prompt": p})
                got += 1
                if got >= want:
                    break
            print(f"{split:5s} {src:10s} {got}/{want}")

    for split, rs in rows.items():
        rng.shuffle(rs)
        with open(os.path.join(a.out_dir, f"{split}.jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"args": vars(a), "sources": manifest,
               "counts": {s: len(r) for s, r in rows.items()}},
              open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
