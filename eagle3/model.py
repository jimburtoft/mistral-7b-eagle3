# SPDX-License-Identifier: Apache-2.0
"""EAGLE3 draft model, device-agnostic (CPU / CUDA / Trainium).

Architecture and checkpoint layout follow the speculators library
(https://github.com/vllm-project/speculators, Apache-2.0, v0.9.0,
``src/speculators/models/eagle3/{core,model_definitions}.py``) so that the saved
checkpoint loads unchanged in vLLM (``method: eagle3``) and in vllm-neuron:

    g   = fc(cat(h_low, h_mid, h_high))                    # 3H -> H
    x   = cat(input_layernorm(embed(tok)), hidden_norm(g)) # 2H
    h   = g' + attn(x)            (g' = hidden_norm(g) when norm_before_residual)
    h   = h + mlp(post_attention_layernorm(h))
    logits = lm_head(norm(h))     (draft vocab)

Differences from speculators (implementation only, same math):
- plain PyTorch attention (no flex_attention / BlockMask, no Triton), so it compiles with
  ``torch.compile(backend="neuron")``; the TTT "diagonal" keys are handled as per-position
  dot products instead of a [S, S*(j+1)] dense mask (same result, O(S^2) memory);
- static shapes: documents are packed into fixed-length rows (block-diagonal causal mask,
  as speculators does), TTT steps are unrolled;
- the TTT KV cache is a Python list of per-step K/V tensors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Eagle3Config:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e6
    max_position_embeddings: int = 32768
    vocab_size: int = 32768
    draft_vocab_size: int = 32768
    aux_layer_ids: list = field(default_factory=lambda: [2, 16, 29])
    norm_before_residual: bool = True
    num_layers: int = 1

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rope_cos_sin(pos: torch.Tensor, head_dim: int, theta: float, dtype):
    """pos: [B, S] int -> cos, sin [B, S, head_dim] (HF Llama 'default' rope, half-rotation layout)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=pos.device, dtype=torch.float32) / head_dim))
    f = pos.float().unsqueeze(-1) * inv  # [B, S, D/2]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x [B, h, S, D]; cos/sin [B, S, D]
    return x * cos.unsqueeze(1) + _rotate_half(x) * sin.unsqueeze(1)


class Eagle3Attention(nn.Module):
    """Attention for one TTT step.

    Step 0 is ordinary causal attention (per document). In step j > 0 a query at position t
    attends to the step-0 keys (causal) plus exactly one key from each later step: the key
    that step produced at the same position t (speculators' "diagonal draft mask").
    Instead of materialising [S, S*(j+1)] scores, the diagonal terms are computed as
    per-position dot products and joined in one softmax. This is mathematically identical to
    the dense mask (tests/parity_speculators.py checks both) and keeps memory O(S^2).
    """

    def __init__(self, c: Eagle3Config, in_dim: int):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(in_dim, c.num_attention_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, c.num_key_value_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, c.num_key_value_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask0, kv_cache: list):
        """x [B,S,in]; mask0 [B,1,S,S] additive (causal, per document); kv_cache: (k, v) of earlier steps."""
        c = self.c
        B, S, _ = x.shape
        rep = c.num_attention_heads // c.num_key_value_heads
        q = self.q_proj(x).view(B, S, c.num_attention_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        kv_cache.append((k, v))
        scale = 1.0 / math.sqrt(c.head_dim)
        k0, v0 = kv_cache[0]
        s0 = torch.matmul(q, k0.transpose(-1, -2)).float() * scale + mask0      # [B,h,S,S]
        if len(kv_cache) == 1:
            p = torch.softmax(s0, dim=-1).to(q.dtype)
            o = torch.matmul(p, v0)
        else:
            sd = torch.stack([(q * kj).sum(-1) for kj, _ in kv_cache[1:]], dim=-1).float() * scale  # [B,h,S,j]
            m = torch.maximum(s0.amax(-1, keepdim=True), sd.amax(-1, keepdim=True))
            e0 = torch.exp(s0 - m)
            ed = torch.exp(sd - m)
            z = e0.sum(-1, keepdim=True) + ed.sum(-1, keepdim=True)
            o = torch.matmul((e0 / z).to(q.dtype), v0)
            vd = torch.stack([vj for _, vj in kv_cache[1:]], dim=-2)                # [B,h,S,j,D]
            o = o + ((ed / z).to(q.dtype).unsqueeze(-1) * vd).sum(-2)
        o = o.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, c: Eagle3Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3FirstLayer(nn.Module):
    def __init__(self, c: Eagle3Config):
        super().__init__()
        self.c = c
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.hidden_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attn = Eagle3Attention(c, 2 * c.hidden_size)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = MLP(c)

    def forward(self, embeds, hidden, cos, sin, mask0, kv_cache):
        residual = hidden
        e = self.input_layernorm(embeds)
        h = self.hidden_norm(hidden)
        if self.c.norm_before_residual:
            residual = h
        x = residual + self.self_attn(torch.cat([e, h], dim=-1), cos, sin, mask0, kv_cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Eagle3Draft(nn.Module):
    """Trainable part of the draft (+ frozen embed / lm_head buffers).

    embed_tokens [V, H] and lm_head [Vd, H] are frozen (as in speculators: copied from the
    target; lm_head = target lm_head rows selected by t2d). Registered as buffers so the
    optimizer never sees them; saved into the checkpoint for vLLM.
    """

    def __init__(self, c: Eagle3Config):
        super().__init__()
        assert c.num_layers == 1, "only the standard 1-layer EAGLE3 draft is implemented"
        self.c = c
        self.fc = nn.Linear(len(c.aux_layer_ids) * c.hidden_size, c.hidden_size, bias=False)
        self.layers = nn.ModuleList([Eagle3FirstLayer(c)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.register_buffer("embed_tokens", torch.zeros(c.vocab_size, c.hidden_size), persistent=False)
        self.register_buffer("lm_head", torch.zeros(c.draft_vocab_size, c.hidden_size), persistent=False)
        self.register_buffer("d2t", torch.zeros(c.draft_vocab_size, dtype=torch.long), persistent=False)
        self.register_buffer("t2d", torch.zeros(c.vocab_size, dtype=torch.bool), persistent=False)

    def set_frozen(self, embed: torch.Tensor, lm_head_full: torch.Tensor, d2t: torch.Tensor, t2d: torch.Tensor):
        self.embed_tokens = embed.detach().clone().to(self.embed_tokens.device)
        self.lm_head = lm_head_full[t2d].detach().clone().to(self.lm_head.device)
        self.d2t = d2t.clone().to(self.d2t.device)
        self.t2d = t2d.clone().to(self.t2d.device)

    def step(self, ids, hidden, cos, sin, mask0, kv_cache):
        """One TTT step. ids [B,S]; hidden [B,S,H] -> (new hidden [B,S,H], logits [B,S,Vd])."""
        emb = F.embedding(ids, self.embed_tokens).to(hidden.dtype)
        h = self.layers[0](emb, hidden, cos, sin, mask0, kv_cache)
        logits = F.linear(self.norm(h), self.lm_head.to(h.dtype))
        return h, logits


def build_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Additive causal mask restricted to the same document. doc_ids [B,S] long, -1 = padding.

    Padding queries attend to themselves (never an all -inf softmax row); their outputs are
    excluded from the loss.
    """
    B, S = doc_ids.shape
    i = torch.arange(S, device=doc_ids.device)
    causal = (i.unsqueeze(1) >= i.unsqueeze(0)).unsqueeze(0)
    same = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    ok = (causal & same & (doc_ids >= 0).unsqueeze(2)) | (i.unsqueeze(1) == i.unsqueeze(0)).unsqueeze(0)
    neg = torch.finfo(torch.float32).min
    return torch.zeros(B, S, S, device=doc_ids.device).masked_fill(~ok, neg).unsqueeze(1)


def eagle3_loss(draft: Eagle3Draft, aux, ids, last_hidden_logits_draft, loss_mask, doc_ids, pos, ttt_steps: int,
                step_decay: float = 1.0, mask0=None, same_doc_only: bool = True):
    """speculators EAGLE3 objective (TTT, forward KL on the draft vocab).

    Already-shifted inputs (speculators `shift_batch`): position t holds
    ids[t] = x[t+1], aux[t] = g[t] (target aux states at t), target logits for x[t+2]
    are `last_hidden_logits_draft[t]` (target logits at position t+1, restricted to the draft vocab),
    loss_mask[t] = l[t+1].

    aux [B,S,3H]; ids [B,S]; last_hidden_logits_draft [B,S,Vd] (fp32 or bf16);
    loss_mask [B,S] bool; doc_ids [B,S] long (document index within the row, -1 = pad);
    pos [B,S] long (per-document positions; after the shift they start at 1).
    same_doc_only: at TTT step j, drop slots whose j-ahead target lies in the next packed document
    (speculators keeps them; it is a handful of tokens per document boundary).
    Returns total loss (sum over steps of token-weighted masked-mean KL) and per-step accuracy counts
    stats[j] = (#slots whose steps 0..j are all correct, #loss slots, #slots correct through j-1).
    """
    c = draft.c
    B, S, _ = aux.shape
    if mask0 is None:
        mask0 = build_mask(doc_ids)
    hidden = draft.fc(aux)
    kv: list = []
    total = hidden.new_zeros((), dtype=torch.float32)
    stats = []
    prev_ok = loss_mask.clone()
    tgt_logp = torch.log_softmax(last_hidden_logits_draft.float(), dim=-1)
    tgt_p = tgt_logp.exp()
    tgt_ids = tgt_logp.argmax(-1)
    cur_ids = ids
    for j in range(ttt_steps):
        cos, sin = rope_cos_sin(pos + j, c.head_dim, c.rope_theta, hidden.dtype)
        hidden, logits = draft.step(cur_ids, hidden, cos, sin, mask0, kv)
        # align: logits at t predict target position t+j (speculators align_for_step)
        lg = logits[:, : S - j] if j else logits
        tp, tlp, ti = tgt_p[:, j:], tgt_logp[:, j:], tgt_ids[:, j:]
        lmb = loss_mask[:, j:]
        if same_doc_only and j:
            lmb = lmb & (doc_ids[:, j:] == doc_ids[:, : S - j])
        lm = lmb.float()
        lp = torch.log_softmax(lg.float(), dim=-1)
        kl = (tp * (tlp - lp)).sum(-1)  # [B, S-j]
        # token-weighted over all rows (= speculators' reduction over one packed row)
        step_loss = (kl * lm).sum() / (lm.sum() + 1e-5)
        total = total + (step_decay ** j) * step_loss
        correct = lg.argmax(-1) == ti
        pc = prev_ok[:, : S - j] if j else prev_ok
        cond_total = pc.float().sum()
        both = pc & correct
        stats.append(torch.stack([(both & (lm > 0)).float().sum(), lm.sum(), cond_total]))
        prev_ok = both if j == 0 else torch.cat([both, prev_ok[:, S - j:]], dim=1)
        # teacher forcing: next step consumes ground-truth tokens shifted by 1+j
        cur_ids = torch.cat([ids[:, 1 + j:], ids.new_zeros(B, 1 + j)], dim=1)
    return total, torch.stack(stats)  # stats [ttt, 3]: correct, n_masked, cond_total
