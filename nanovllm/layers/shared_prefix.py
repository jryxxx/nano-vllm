"""Paged, single-level shared-prefix decode. No cache ownership is changed.

The caller proves that the prefix pages in every active row are identical.
Lengths and page contents are runtime inputs, including during graph replay.
"""
import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_with_kvcache


@triton.jit
def _prefix(Q, K, V, BT, LENGTHS, PREFIX, OUT, LSE,
            B: tl.constexpr, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
            PAGE: tl.constexpr, QB: tl.constexpr, QH: tl.constexpr,
            KP: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr,
            VP: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr,
            SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    kh = tl.program_id(1)
    b, h = r // (H // HK), kh * (H // HK) + r % (H // HK)
    d = tl.arange(0, D)
    active = tl.load(LENGTHS + b, b < B, other=0) > 0
    q = tl.load(Q + b[:, None] * QB + h[:, None] * QH + d[None, :],
                (b[:, None] < B) & active[:, None], other=0)
    p_len = tl.load(PREFIX)
    m = tl.full((BM,), -float('inf'), tl.float32)
    z = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, D), 0, tl.float32)
    for start in range(tl.cdiv(p_len, BN)):
        n = start * BN + tl.arange(0, BN)
        page = tl.load(BT + n // PAGE, n < p_len, other=0)
        k = tl.load(K + page[None, :] * KP + (n[None, :] % PAGE) * KT
                    + kh * KH + d[:, None], n[None, :] < p_len, other=0)
        v = tl.load(V + page[:, None] * VP + (n[:, None] % PAGE) * VT
                    + kh * VH + d[None, :], n[:, None] < p_len, other=0)
        scores = tl.dot(q, k) * SCALE
        scores = tl.where(n[None, :] < p_len, scores, -float('inf'))
        nm = tl.maximum(m, tl.max(scores, 1))
        prob = tl.exp(scores - nm[:, None])
        alpha = tl.exp(m - nm)
        acc = acc * alpha[:, None] + tl.dot(prob.to(v.dtype), v)
        z = z * alpha + tl.sum(prob, 1)
        m = nm
    valid = active & (p_len > 0)
    out = tl.where(valid[:, None], acc / tl.maximum(z[:, None], 1), 0)
    lse = tl.where(valid, m + tl.log(tl.maximum(z, 1)), -float('inf'))
    tl.store(OUT + (b[:, None] * H + h[:, None]) * D + d[None, :], out, b[:, None] < B)
    tl.store(LSE + b * H + h, lse, b < B)


@triton.jit
def _merge(A, LA, C, LC, SUFFIX_LENGTHS, LENGTHS, PREFIX, OUT,
           N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, b = x // D, x // (H * D)
    active = tl.load(LENGTHS + b, x < N, other=0) > 0
    vp = active & (tl.load(PREFIX) > 0)
    vs = active & (tl.load(SUFFIX_LENGTHS + b, x < N, other=0) > 0)
    a = tl.load(LA + row, x < N, other=0)
    c = tl.load(LC + row, x < N, other=0)
    a, c = tl.where(vp, a, -float('inf')), tl.where(vs, c, -float('inf'))
    m = tl.where(vp | vs, tl.maximum(a, c), 0)
    wa, wc = tl.where(vp, tl.exp(a - m), 0), tl.where(vs, tl.exp(c - m), 0)
    av = tl.where(vp, tl.load(A + x, x < N, other=0), 0)
    cv = tl.where(vs, tl.load(C + x, x < N, other=0), 0)
    tl.store(OUT + x, (wa * av + wc * cv) / tl.maximum(wa + wc, 1), x < N)


def shared_prefix_decode(q, k, v, block_tables, lengths, prefix_length,
                         suffix_tables, suffix_lengths, scale):
    """q: [batch, heads, dim]; paged KV; full-block prefix; arbitrary suffixes.

    prefix_length is a one-element int32 CUDA tensor. Zero-length suffixes and
    inactive graph-padding rows are defined to contribute zero attention mass.
    """
    b, h, d = q.shape
    hk = k.shape[2]
    op = torch.empty((b, h, d), dtype=torch.float32, device=q.device)
    lp = torch.empty((b, h), dtype=torch.float32, device=q.device)
    _prefix[(triton.cdiv(b * (h // hk), 16), hk)](
        q, k, v, block_tables, lengths, prefix_length, op, lp,
        b, h, hk, d, k.shape[1], q.stride(0), q.stride(1),
        *k.stride()[:3], *v.stride()[:3], scale, 16, 64,
        num_warps=4, num_stages=1)
    os, ls = flash_attn_with_kvcache(q.unsqueeze(1), k, v,
                                    cache_seqlens=suffix_lengths, block_table=suffix_tables,
                                    softmax_scale=scale, causal=True, return_softmax_lse=True)
    out = torch.empty((b, h, d), dtype=q.dtype, device=q.device)
    _merge[(triton.cdiv(out.numel(), 256),)](
        op, lp, os, ls, suffix_lengths, lengths, prefix_length, out,
        out.numel(), h, d, 256)
    return out.unsqueeze(1)
