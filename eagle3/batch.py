# SPDX-License-Identifier: Apache-2.0
"""Shifted EAGLE3 training layout (speculators ``shift_batch``, applied per packed document).

For a packed row with source tokens x, aux states g, target final states y and loss mask l,
slot k with source position t = gi[k] holds:
    ids[k] = x[t+1]   aux[k] = g[t]   final[k] = y[t+1] (predicts x[t+2])
    lm[k]  = l[t+1]   pos[k] = pos[t+1]   doc[k] = doc[t]
Padding slots have gi = -1, doc = -1, lm = 0.
"""

from __future__ import annotations

import torch


def shifted_cpu(ids, doc, pos, lm, gi):
    """CPU tensors [B,S] -> shifted ids/doc/pos/lm [B,S] + clamped gather index."""
    valid = gi >= 0
    g = gi.clamp(min=0).long()
    g1 = (g + 1).clamp(max=ids.shape[1] - 1)
    take = lambda x, i: torch.gather(x, 1, i)
    ids_s = torch.where(valid, take(ids, g1), 0)
    doc_s = torch.where(valid, take(doc, g), -1)
    pos_s = torch.where(valid, take(pos, g1), 0)
    lm_s = valid & (take(lm, g1) > 0)
    return ids_s.long(), doc_s.long(), pos_s.long(), lm_s, g, g1


def gather_rows(x, idx):
    """x [B,S,D], idx [B,S] long -> x[b, idx[b, s]]."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
