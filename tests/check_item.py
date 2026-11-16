# SPDX-License-Identifier: Apache-2.0
"""Check a producer queue item (computed on an accelerator) against a CPU fp32 recompute.

Usage: python tests/check_item.py <item.pt> <rows glob> <target dir> [--layers N]
Recomputes the same row on CPU (fp32, functional target) and reports the cosine and relative
error of aux / final states, and the argmax agreement of target logits on loss tokens.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eagle3.batch import gather_rows, shifted_cpu  # noqa: E402
from eagle3.target import Target  # noqa: E402
from train.producer import load_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("item")
    ap.add_argument("rows")
    ap.add_argument("target")
    ap.add_argument("--aux-layers", default="2,16,29")
    a = ap.parse_args()
    it = torch.load(a.item, weights_only=False)
    rows = load_rows(a.rows)
    # find the row by content
    ids_all = rows["ids"].long()
    gi = rows["gi"]
    match = None
    for r in range(ids_all.shape[0]):
        s = shifted_cpu(ids_all[r:r + 1], rows["doc"][r:r + 1].long(), rows["pos"][r:r + 1].long(),
                        rows["lm"][r:r + 1], gi[r:r + 1])[0]
        if torch.equal(s, it["ids"][:1]):
            match = r
            break
    assert match is not None, "row not found"
    tg = Target(a.target, [int(x) for x in a.aux_layers.split(",")], torch.device("cpu"), dtype=torch.float32)
    r = slice(match, match + 1)
    _, _, _, lm_s, g, g1 = shifted_cpu(ids_all[r], rows["doc"][r].long(), rows["pos"][r].long(), rows["lm"][r], gi[r])
    aux, final = tg.features(ids_all[r], rows["doc"][r].long(), rows["pos"][r].long())
    aux, final = gather_rows(aux, g), gather_rows(final, g1)
    m = lm_s[0]
    for name, ref, got in (("aux", aux[0], it["aux"][0].float()), ("final", final[0], it["final"][0].float())):
        ref, got = ref[m], got[m]
        H = tg.c.hidden_size
        for k in range(ref.shape[-1] // H):
            rr, gg = ref[:, k * H:(k + 1) * H], got[:, k * H:(k + 1) * H]
            cos = torch.nn.functional.cosine_similarity(rr, gg, dim=-1)
            rel = (rr - gg).norm() / rr.norm()
            print(f"{name}[{k}] cos min {cos.min():.5f} mean {cos.mean():.6f} rel {rel:.4f}")
    lm_w = tg.lm_head.float()
    a1 = (final[0][m] @ lm_w.T).argmax(-1)
    a2 = (it["final"][0].float()[m] @ lm_w.T).argmax(-1)
    print(f"target argmax agreement on {int(m.sum())} loss tokens: {(a1 == a2).float().mean():.4f}")


if __name__ == "__main__":
    main()
