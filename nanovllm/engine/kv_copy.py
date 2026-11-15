"""Copy only computed KV tokens; identical implementation for COW and full-copy forks."""
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_pages(cache, jobs, PAGE: tl.constexpr, PLANE: tl.constexpr,
                WIDTH: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    plane, job = tl.program_id(1).to(tl.int64), tl.program_id(2)
    src = tl.load(jobs + job * 3)
    dst = tl.load(jobs + job * 3 + 1)
    valid = tl.load(jobs + job * 3 + 2) * WIDTH
    values = tl.load(cache + plane * PLANE + src * PAGE + offsets, offsets < valid, 0)
    tl.store(cache + plane * PLANE + dst * PAGE + offsets, values, offsets < valid)


@torch.inference_mode()
def copy_kv_pages(cache, jobs):
    if not jobs:
        return
    metadata = torch.tensor(jobs, dtype=torch.int64, device=cache.device)
    width = cache.shape[-2] * cache.shape[-1]
    try:
        _copy_pages[(triton.cdiv(max(j[2] for j in jobs) * width, 1024),
                     cache.shape[0] * cache.shape[1], len(jobs))](
            cache, metadata, cache.stride(2), cache.stride(1), width, 1024)
    finally:
        # Hold source/destination references even if launch or completion fails.
        torch.cuda.synchronize(cache.device)
