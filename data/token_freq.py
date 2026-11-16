# SPDX-License-Identifier: Apache-2.0
"""Token frequencies of assistant (loss-masked) tokens over packed rows -> token_freq.pt.

Same statistic speculators uses to choose the draft vocabulary.
"""

import argparse
import glob

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cnt = None
    for f in sorted(glob.glob(a.rows)):
        z = np.load(f)
        ids = z["ids"][z["lm"] > 0]
        c = np.bincount(ids, minlength=1)
        if cnt is None:
            cnt = c
        else:
            if len(c) > len(cnt):
                cnt = np.pad(cnt, (0, len(c) - len(cnt)))
            cnt[: len(c)] += c
    freq = {int(i): int(n) for i, n in enumerate(cnt) if n > 0}
    torch.save(freq, a.out)
    print(f"{len(freq)} distinct assistant tokens, {int(cnt.sum())} total")


if __name__ == "__main__":
    main()
