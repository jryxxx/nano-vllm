# KV Cache sessions and copy-on-write branches

This phase builds on the upstream-derived remote checkout and its paged shared-prefix attention implementation. `nano-vllm-pre-cow` preserves the implementation before this phase. Original upstream remains untouched.

## Semantics

A branch point is a completed engine step: `num_cached_tokens == len(token_ids) - 1`, with no scheduled work. The last sampled token is retained in the history but has not yet been evaluated into KV. A fork treats the entire inherited history as its prompt and starts a new output-token budget. `next_token_id` replaces only that pending token; `append_token_ids` adds a continuation after it. Already-computed tokens cannot be edited.

A saved session is an immutable snapshot of token history, cached length and page IDs. It survives completion or cancellation of its source request. Saving the same name acquires the new snapshot's references before releasing the old snapshot. `close_session` releases snapshot ownership; it does not cancel descendants. Request cancellation and session closure are idempotent.

Full pages and the computed portion of a partial last page are initially shared. Before a decode or prefill write, the allocator counts both missing pages and shared writable pages. It reserves enough free pages, copies only computed tokens from a shared writable page, and commits the new mapping after the copy completes. At a full-page boundary a fresh empty page is allocated without a tail copy. Full-copy mode duplicates all computed pages with the same copy kernel and is the comparison baseline.

## Ownership and failure behavior

`Block.ref_count` counts live request and snapshot owners. The allocator keeps an owner ledger and exposes `assert_invariants()` to verify exact counts, the free/used partition and hash-index consistency. A zero-reference page returns to the free deque; valid full-page contents may remain as an evictable prefix-cache entry, preserving upstream behavior.

Copies are synchronous transactions at step boundaries. Source owners retain their references and new destinations hold an allocation reference until GPU synchronization completes. There are no deferred copy descriptors or unpinned transfers, so a source or destination cannot be recycled while copying. A failed copy releases only provisional destination pages; original mappings remain intact. The engine restores scheduling state after a scheduling/copy exception. If model execution raises, scheduled requests lose their potentially partially written cache and return to prefill. A fatal CUDA context error still requires restarting the process; injected recoverable failures test allocator rollback, not recovery from a poisoned device context.

Retained sessions are never silently evicted. If no request can run because live owners hold all pages, scheduling raises `CacheCapacityError`. The caller can close sessions or cancel requests and retry. There is no automatic session TTL, CPU offload, distributed ownership or asynchronous copy overlap in this phase. Session mutation is a single-thread, single-GPU API; tensor-parallel engines reject it explicitly.

## Public API

```python
from nanovllm import LLM, SamplingParams

llm = LLM(model_path, max_model_len=8192)
request = llm.add_request(prompt_tokens, SamplingParams(max_tokens=64))
llm.step()  # ensure prefill has completed; long chunked prompts may need more steps
llm.pause_request(request)
llm.save_session(request, "checkpoint")

branches = llm.fork_requests(
    request, count=8,
    sampling_params=SamplingParams(max_tokens=32),
    next_token_ids=[100, 101, 102, 103, 104, 105, 106, 107],
    cache_mode="cow",  # "copy" is the full-copy baseline
)
llm.cancel_request(request)
while not llm.is_finished():
    completed, processed = llm.step()
llm.close_session("checkpoint")
assert llm.cache_stats()["live_pages"] == 0
```

Additional calls: `fork_request`, `fork_session`, `resume_request`, `cancel_request`, `save_session`, `close_session`, `cache_stats`. `add_request(..., session_id="name")` saves a snapshot automatically when generation completes. A fork can similarly specify `session_id` or `output_session_id`. Bulk forks cannot share a single output session name. Retained or paused owners do not make `is_finished()` false: it describes runnable work, not whether every cache reference has been closed.

The existing GPU KV pool remains preallocated. `live_pages` measures unique physical pages with live owners, not a reduction in `torch.cuda.memory_reserved()`. A page becoming free makes it reusable within that pool; it is not returned to the CUDA allocator.

## Implementation map

- `engine/block_manager.py`: snapshot ownership, bulk clone transaction, write planning, tail COW, refcount release, invariant checker and occupancy/copy counters.
- `engine/kv_copy.py`: batched Triton page copy, valid-token masks, 64-bit plane addressing and completion synchronization.
- `engine/scheduler.py`: request registry, pause/resume/cancel, named snapshots, COW before prefill/decode, automatic snapshot on completion and explicit capacity failure.
- `engine/llm_engine.py`: validated branch APIs, step-boundary guard, copy callback and exception recovery.
- `tests/validate_cow_metadata.py`: deterministic rollback cases and 2,000 random ownership operations.
- `tests/validate_cow_gpu.py`: page boundaries, partial-tail divergence, continuation prefill, full-recompute oracle, multi-generation sessions, cancellation, snapshot replacement, injected copy failure, constrained capacity and shared-prefix graph integration.
- `experiments/bench_cow.py`: paired bulk COW/full-copy trials using the same model, kernel, pending tokens and fixed output lengths.

## Reproduction on this server

```bash
bash /root/autodl-tmp/shared-context/run_cow_env.sh python tests/validate_cow_metadata.py
bash /root/autodl-tmp/shared-context/run_cow_env.sh python tests/validate_cow_gpu.py \
  --model /root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16 \
  --backend auto --output results/cow/gpu-auto.json
bash /root/autodl-tmp/shared-context/run_cow_env.sh python experiments/bench_cow.py \
  --model /root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16 \
  --output results/cow/benchmark.json
```

Benchmarks exclude parent creation/prefill, which is identical and precedes the branch point. They include CPU branch construction, the full-copy transaction, deferred COW on the first write, 32 output tokens per branch, and normal completion/reclamation. Both modes use ordinary attention (`shared_attention_backend="off"`) to isolate cache sharing from the earlier custom shared-attention kernel. One warmup pair is excluded per scenario, three measured pairs alternate order. Report medians and ranges, not statistical significance from three observations. Hardware performance counters are unavailable; no DRAM-transaction or cache-hit-rate claims are inferred from timing.
