# SPDX-License-Identifier: Apache-2.0
"""EAGLE3 draft trainer. Runs on Trainium (``neuron``), CUDA or CPU (``--device``).

Consumes shifted items from the producer queue (train/producer.py) or, with ``--inline``,
computes target features in-process (single device that fits target + draft, e.g. a GPU, or a
CPU smoke test with a truncated target).

Objective: speculators EAGLE3 (TTT steps, forward KL on the draft vocab; eagle3/model.py).
Optimizer: AdamW (fp32 master weights) with bf16 autocast, linear warmup + cosine decay,
grad clip 1.0. The frozen embedding and lm_head are buffers, not parameters.

Default layout on one trn2.3xlarge (LNC=2, 4 x 24 GB logical cores): producers on cores 0-2,
this trainer on core 3 (see train/launch.sh).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eagle3.checkpoint import load_speculators, save_speculators  # noqa: E402
from eagle3.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from eagle3.model import Eagle3Config, Eagle3Draft, eagle3_loss  # noqa: E402


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def load_head(target_dir):
    """embed [V,H], lm_head [V,H] (bf16, CPU) from the target safetensors."""
    from safetensors import safe_open
    idx_f = os.path.join(target_dir, "model.safetensors.index.json")
    wm = json.load(open(idx_f))["weight_map"] if os.path.exists(idx_f) else None
    out = {}
    for short, key in (("embed", "model.embed_tokens.weight"), ("lm_head", "lm_head.weight")):
        f = wm[key] if wm else "model.safetensors"
        with safe_open(os.path.join(target_dir, f), "pt") as fh:
            out[short] = fh.get_tensor(key).to(torch.bfloat16)
    return out["embed"], out["lm_head"]


def build_vocab(freq_path, vocab_size, draft_vocab_size):
    """speculators build_vocab_mappings_from_distribution: top-k assistant-token frequencies."""
    if draft_vocab_size >= vocab_size:
        return torch.zeros(vocab_size, dtype=torch.long), torch.ones(vocab_size, dtype=torch.bool)
    freq = torch.load(freq_path)
    order = sorted(freq, key=lambda t: (-freq[t], t))[:draft_vocab_size]
    sel = sorted(order)
    i = 0
    while len(sel) < draft_vocab_size:  # pad with low ids if the data has fewer distinct tokens
        if i not in freq:
            sel.append(i)
        i += 1
    sel = torch.tensor(sorted(sel), dtype=torch.long)
    t2d = torch.zeros(vocab_size, dtype=torch.bool)
    t2d[sel] = True
    return sel - torch.arange(draft_vocab_size), t2d


class Step(torch.nn.Module):
    """One forward of the objective (compiled as one graph)."""

    def __init__(self, draft: Eagle3Draft, ttt: int, bf16: bool):
        super().__init__()
        self.draft, self.ttt, self.bf16 = draft, ttt, bf16

    def forward(self, aux, final, ids, lm, doc, pos):
        d = self.draft
        # target logits on the draft vocab (speculators: verifier_lm_head[t2d] @ verifier_norm(h_last));
        # `final` is already the target's final-norm output
        tgt_logits = torch.nn.functional.linear(final, d.lm_head.to(final.dtype))
        if self.bf16:
            with torch.autocast(device_type=aux.device.type, dtype=torch.bfloat16):
                return eagle3_loss(d, aux, ids, tgt_logits, lm, doc, pos, self.ttt)
        return eagle3_loss(d, aux.float(), ids, tgt_logits, lm, doc, pos, self.ttt)


def to_dev(it, dev):
    return [it[k].to(dev) for k in ("aux", "final", "ids", "lm", "doc", "pos")]


def acc_summary(st):
    """st [ttt,3] summed counts -> per-step conditional acc + expected acceptance length (k=ttt)."""
    st = st.double()
    full = st[:, 0] / st[:, 1].clamp(min=1)  # P(steps 0..j all correct)
    cond = [float(st[0, 0] / st[0, 1].clamp(min=1))] + \
           [float(st[j, 0] / st[j - 1, 0].clamp(min=1)) for j in range(1, st.shape[0])]
    return {"acc_full": [round(float(x), 4) for x in full], "acc_cond": [round(x, 4) for x in cond],
            "accept_len": round(1 + float(full.sum()), 4)}


@torch.no_grad()
def evaluate(step_fn, eval_dir: Path, dev):
    tot = None
    loss_sum, n = 0.0, 0
    for f in sorted(eval_dir.glob("r*_*.pt")):
        it = torch.load(f, weights_only=False)
        loss, st = step_fn(*to_dev(it, dev))
        st = st.float().cpu()
        tot = st if tot is None else tot + st
        loss_sum += float(loss)
        n += 1
    r = acc_summary(tot)
    r["loss"] = round(loss_sum / max(n, 1), 5)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target checkpoint dir (embed / lm_head / config)")
    ap.add_argument("--verifier-name", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--queue", default="/dev/shm/e3q")
    ap.add_argument("--eval-dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--ttt", type=int, default=3)
    ap.add_argument("--aux-layers", default="2,16,29")
    ap.add_argument("--draft-vocab", type=int, default=32768)
    ap.add_argument("--token-freq", default="")
    ap.add_argument("--compute", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--init", default="", help="speculators-format checkpoint dir to start from")
    ap.add_argument("--resume", action="store_true",
                    help="with --init: also load <init>/optim.pt and continue the LR schedule at --start-step")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--keep-queue", action="store_true")
    ap.add_argument("--max-items", type=int, default=0, help="debug: stop after N queue items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compile", action="store_true")
    add_device_arg(ap)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = get_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    tcfg = json.load(open(os.path.join(a.target, "config.json")))
    H = tcfg["hidden_size"]
    nh = tcfg["num_attention_heads"]
    c = Eagle3Config(hidden_size=H, intermediate_size=tcfg["intermediate_size"], num_attention_heads=nh,
                     num_key_value_heads=tcfg["num_key_value_heads"], head_dim=tcfg.get("head_dim") or H // nh,
                     rms_norm_eps=tcfg["rms_norm_eps"], rope_theta=tcfg.get("rope_theta", 1e6),
                     max_position_embeddings=tcfg["max_position_embeddings"], vocab_size=tcfg["vocab_size"],
                     draft_vocab_size=min(a.draft_vocab, tcfg["vocab_size"]),
                     aux_layer_ids=[int(x) for x in a.aux_layers.split(",")])
    if a.init:
        draft = load_speculators(a.init)
        assert draft.c.aux_layer_ids == c.aux_layer_ids
    else:
        draft = Eagle3Draft(c)
        for m in draft.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
        embed, lm_head = load_head(a.target)
        d2t, t2d = build_vocab(a.token_freq, c.vocab_size, c.draft_vocab_size)
        draft.set_frozen(embed, lm_head, d2t, t2d)
    draft = draft.to(dev)
    # fp32 master weights for trainable params, frozen tables in bf16
    for p in draft.parameters():
        p.data = p.data.float()
    draft.embed_tokens = draft.embed_tokens.to(torch.bfloat16)
    draft.lm_head = draft.lm_head.to(torch.bfloat16)

    mod = Step(draft, a.ttt, a.compute == "bf16").to(dev)
    step_fn = maybe_compile(mod, dev, enabled=not a.no_compile)
    params = [p for p in draft.parameters() if p.requires_grad]
    print(json.dumps({"trainable_params": sum(p.numel() for p in params), "device": str(dev)}), flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    if a.resume:
        st = torch.load(os.path.join(a.init, "optim.pt"), map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and k != "step":
                    state[k] = v.to(dev)
    q = Path(a.queue)
    eval_dir = Path(a.eval_dir) if a.eval_dir else None
    log = open(out / "train.jsonl", "a")
    seen = set()

    def next_item():
        while True:
            files = sorted(q.glob("e*_r*_*.pt"), key=lambda p: p.stat().st_mtime)
            files = [f for f in files if f.name not in seen] if a.keep_queue else files
            if files:
                f = files[0]
                try:
                    it = torch.load(f, weights_only=False)
                except Exception:
                    time.sleep(0.05)
                    continue
                if a.keep_queue:
                    seen.add(f.name)
                else:
                    f.unlink(missing_ok=True)
                return it
            time.sleep(0.02)

    def do_eval(s):
        if eval_dir is None:
            return
        draft.eval()
        t = time.time()
        r = evaluate(step_fn, eval_dir, dev)
        draft.train()
        rec = {"eval_step": s, "sec": round(time.time() - t, 1), **r}
        print("EVAL " + json.dumps(rec), flush=True)
        with open(out / "eval.jsonl", "a") as f:
            print(json.dumps(rec), file=f)

    def save(tag):
        save_speculators(draft, str(out / tag), a.verifier_name, tcfg, ttt_steps=a.ttt)
        torch.save({"opt": opt.state_dict(), "step": tag}, out / tag / "optim.pt")

    do_eval(a.start_step)
    t0, wait, n_tok, items = time.time(), 0.0, 0, 0
    st_acc = None
    for step in range(a.start_step, a.total_steps):
        lr = lr_at(step, a.total_steps, a.lr, a.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tl = 0.0
        for _ in range(a.accum):
            tw = time.time()
            it = next_item()
            wait += time.time() - tw
            loss, st = step_fn(*to_dev(it, dev))
            (loss / a.accum).backward()
            items += 1
            n_tok += int(it["lm"].sum())
            if (step + 1) % a.log_every == 0 or step == 0:
                tl += float(loss.detach()) / a.accum
                stc = st.detach().float().cpu()
                st_acc = stc if st_acc is None else st_acc + stc
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % a.log_every == 0 or step == 0:
            el = time.time() - t0
            rec = {"step": step + 1, "loss": round(tl, 5), "grad_norm": round(float(gn), 4), "lr": lr,
                   **acc_summary(st_acc), "sec": round(el, 1), "step_s": round(el / (step + 1 - a.start_step), 3),
                   "queue_wait_frac": round(wait / max(el, 1e-9), 3), "loss_tok_per_s": round(n_tok / el, 1)}
            st_acc = None
            print(json.dumps(rec), file=log, flush=True)
            print(json.dumps(rec), flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.total_steps:
            do_eval(step + 1)
        if (step + 1) % a.save_every == 0 or step + 1 == a.total_steps:
            save(f"step{step + 1:07d}")
        if a.max_items and items >= a.max_items:
            save(f"step{step + 1:07d}")
            break


if __name__ == "__main__":
    main()
