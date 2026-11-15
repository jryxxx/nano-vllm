# Prefix cache eviction policies

This stage extends the remote nano-vLLM development branch. The frozen predecessor is `../nano-vllm-pre-cache-policy`; it includes the shared-prefix attention, session COW, and predictive Prefill scheduler stages.

## Configuration

`cache_policy='fifo'` remains the default. Alternatives are `lru`, `leaf_lru`, and `cost`. `cache_half_life=32` is measured in actual sequence allocations, including re-admission after preemption, not seconds. `schedule_cost_profile` supplies the existing Prefill calibration; without it, cost policy uses a token/attention-work proxy. No new policy is automatically enabled.

## Ownership and identity

Only zero-reference pages in the free list are candidates. Active requests and saved sessions protect their pages through the existing reference ledger. An immutable, interned `(parent identity, exact block token tuple)` chain controls prefix lookup; the legacy 64-bit hash remains diagnostic metadata and cannot cause a false hit under collision. Prefix identities carry physical locations and weak child links. Parent references preserve lineage while descendants remain resident; weak interning releases unused identities. Slot generations increase on reuse and support invariant/debug checks; lookup uses live node locations rather than a stale slot descriptor.

Complete-page clones publish identities only after the synchronous GPU copy succeeds. Failed transfers release destinations without publishing reusable cache contents. Partial pages are never prefix hits. The pool size is fixed; policies never evict live owners or offload to CPU.

## Victim selection

FIFO preserves the predecessor's free-list order. Other policies first reuse unpublished/partial pages, then redundant physical copies of the same logical prefix. LRU then evicts the least recently executed/reused prefix; ties prefer deeper pages. `leaf_lru` additionally prefers identities with no surviving descendants, protecting useful prefix chains. If no free leaf exists, it falls back to free pages so metadata constraints cannot cause a false out-of-memory condition.

`cost` uses the same leaf candidates, with score

`(1 + decayed reuse count) × marginal Prefill work × access-age decay`.

Both decays use the configured half-life. Marginal work uses the calibrated linear token and attention terms for one block at its prefix depth. Fixed launch costs and empirical model residuals are excluded; the score is a retention heuristic, not a measured latency saving. Actual allocations increment reuse counts. Execution touches recency; speculative `can_allocate` calls do neither. No future requests are used.

Victim selection currently scans free pages. `policy_seconds` records selector time only, excluding lookup, exact identity bookkeeping and GPU execution. This implementation targets the single-GPU prototype and does not establish large-pool scaling.

## Experiment interpretation

`experiments/bench_cache.py` replays formatted synthetic document requests in identical order for every policy: hot-prefix pollution by one-off long requests, a midpoint hotspot switch, a no-reuse stream, and mixed prefix lengths. Each request generates four tokens. The experiment is sequential and closed-loop, so reported throughput is not concurrent serving goodput or an arrival-rate capacity measurement.

The logical block manager is limited to 20/32/64 pages while the physical GPU KV tensor remains unchanged. Capacity comparisons therefore quantify cache competition, not reductions in reserved GPU memory. Shared attention stays off, scheduling stays baseline, and the same calibrated cost profile is supplied to all policies.

Correctness checks separately cover CPU ownership/identity, bitwise resident KV preservation, COW regression, and logits against full recomputation. Full recomputation changes Prefill segmentation and can introduce FP16 numerical differences. A failed historical gate remains a failed gate even when the predecessor also fails it; it must not be relabeled as a fully validated quality result.
