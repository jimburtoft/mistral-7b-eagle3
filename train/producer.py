# SPDX-License-Identifier: Apache-2.0
"""Target-feature producer for EAGLE3 training (one process per accelerator core / device).

Reads packed rows (data/pack.py), runs the frozen target, and writes shifted training items
(eagle3/batch.py layout) into a bounded file queue (default /dev/shm). Each item:
  ids, doc, pos [R,S] long; lm [R,S] bool; aux [R,S,3H] bf16; final [R,S,H] bf16
The trainer computes target logits from `final` with the frozen lm_head.

Producer k owns rows i % world == k; rows are reshuffled each epoch. With --eval-out it
writes every row of --rows once (fixed eval items) and exits.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eagle3.batch import gather_rows, shifted_cpu  # noqa: E402
from eagle3.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from eagle3.target import Target  # noqa: E402


def load_rows(pattern):
    keys = ("ids", "doc", "pos", "lm", "gi")
    parts = {k: [] for k in keys}
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        for k in keys:
            parts[k].append(z[k])
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}


def make_item(tg, rows, idx, dev):
    ids, doc, pos, lm, gi = (rows[k][idx] for k in ("ids", "doc", "pos", "lm", "gi"))
    ids_s, doc_s, pos_s, lm_s, g, g1 = shifted_cpu(ids.long(), doc.long(), pos.long(), lm, gi)
    aux, final = tg.features(ids.long().to(dev), doc.long().to(dev), pos.long().to(dev))
    # Reorder on CPU: an element-wise gather with an expanded index is slow to compile on Neuron.
    aux_s = gather_rows(aux.cpu(), g)
    final_s = gather_rows(final.cpu(), g1)
    return {"ids": ids_s, "doc": doc_s, "pos": pos_s, "lm": lm_s, "aux": aux_s, "final": final_s}


def put(q: Path, name: str, item):
    tmp = q / ("." + name + ".tmp")
    torch.save(item, tmp)
    os.replace(tmp, q / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="glob of packed row shards (rows_*.npz)")
    ap.add_argument("--target", required=True, help="local target checkpoint dir (HF safetensors)")
    ap.add_argument("--aux-layers", default="2,16,29")
    ap.add_argument("--queue", default="/dev/shm/e3q")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--rows-per-item", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-queue", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-out", default="")
    ap.add_argument("--max-layers", type=int, default=0, help="debug: truncate the target")
    ap.add_argument("--no-compile", action="store_true")
    add_device_arg(ap)
    a = ap.parse_args()

    dev = get_device(a.device)
    rows = load_rows(a.rows)
    n = rows["ids"].shape[0]
    tg = Target(a.target, [int(x) for x in a.aux_layers.split(",")], dev,
                compile_fn=None if a.no_compile else (lambda f: maybe_compile(f, dev)),
                max_layers=a.max_layers or None)
    R = a.rows_per_item

    if a.eval_out:
        eo = Path(a.eval_out)
        eo.mkdir(parents=True, exist_ok=True)
        mine = list(range(a.rank, n, a.world))
        for s0 in range(0, len(mine), R):
            idx = mine[s0:s0 + R]
            n_real = len(idx)
            idx = idx + [idx[0]] * (R - n_real)
            it = make_item(tg, rows, torch.tensor(idx), dev)
            it["lm"][n_real:] = False
            put(eo, f"r{a.rank}_{s0 // R:06d}.pt", it)
        print(json.dumps({"rank": a.rank, "eval_items": (len(mine) + R - 1) // R}), flush=True)
        return

    q = Path(a.queue)
    q.mkdir(parents=True, exist_ok=True)
    mine = torch.arange(a.rank, n, a.world)
    log = open(q.parent / f"{q.name}_producer{a.rank}.log", "a")
    t0, done = time.time(), 0
    for ep in range(a.epochs):
        g = torch.Generator().manual_seed(a.seed * 100003 + ep * 101 + a.rank)
        order = mine[torch.randperm(len(mine), generator=g)]
        for k in range(0, len(order) - R + 1, R):
            while len(list(q.glob(f"e*_r{a.rank}_*.pt"))) >= a.max_queue:
                time.sleep(0.05)
            it = make_item(tg, rows, order[k:k + R], dev)
            it.update({"epoch": ep, "producer": a.rank})
            put(q, f"e{ep:03d}_r{a.rank}_{k // R:07d}.pt", it)
            done += 1
            if done % 20 == 0:
                print(json.dumps({"rank": a.rank, "epoch": ep, "done": done,
                                  "items_per_s": round(done / (time.time() - t0), 3)}), file=log, flush=True)


if __name__ == "__main__":
    main()
