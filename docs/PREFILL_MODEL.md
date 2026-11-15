# Measured Prefill costs and joint stage planning

This phase keeps the previous scheduler available through its original fixed-chunk settings. The frozen comparison checkout is `/root/autodl-tmp/shared-context/src/nano-vllm-pre-prefill-cost`; original upstream remains untouched. The default policy is still `baseline`.

## Cost model

Calibration times the complete runner Prefill path: input/block metadata preparation, model execution and sampling synchronization. Cached prefixes are materialized by the actual model before measurement. Each shape has two discarded warmups and five recorded timings. These measurements are separate from online workload results.

The non-negative affine backbone uses five features:

- an intercept for fixed execution costs;
- whether any query reads an existing cached prefix;
- total query tokens / 1024;
- the sum of `q * cached + q * (q + 1) / 2`, scaled by 2^20;
- batch size minus one.

This backbone alone is insufficient for all kernel-shape transitions. Exact calibrated shapes use their measured median. Other supported shapes use the affine prediction plus a distance-weighted residual from the three nearest training shapes in query-token, batch-size and query-weighted cached-context space. Holdout and fresh-audit rows are never added to this lookup.

The measured domain is batch <= 8, total query tokens <= 8192 and cached context <= 3072. Predictions outside that domain, or too far from any training neighbor, return unknown rather than extrapolating. The caller falls back to the previous bounded chunk policy. The model/dtype/GPU profile is specific to the recorded environment; it must be recalibrated elsewhere. A bounded scalar EWMA corrects runner-to-engine and recent timing differences; the factor stays between 0.8 and 1.5.

The calibrated backbone may set some coefficients to zero; this is an empirical fit, not a claim that batch structure or the cached path has no performance effect. The exact-shape table and residual lookup still distinguish their measurements.

## Planning

The planner previews the existing waiting order without allocating or changing KV ownership. It considers token budgets 256, 512, 1024, 2048, 3072, 4096, 6144 and 8192, bounded by the engine token budget, then deduplicates identical resulting batch shapes. Capacity and prefix-hit results are read through the existing block manager. Actual allocation remains transactional through the existing prefill/COW path; a shortened batch under memory pressure is predicted again before recording.

For each candidate, it estimates the Prefill time and reserves time for oldest-first decode rounds after it. A candidate is feasible when every current decode round's service age plus estimated completion time fits the configurable soft service budget. Unknown decode shapes use a conservative larger ordinary-attention calibration shape; if no bound exists, the planner falls back.

Among feasible candidates, the planner chooses the largest amount of query work, favoring prompt completion and amortizing fixed execution overhead. If no decode requests are runnable, it can batch more waiting prefill work without paying for unnecessary small chunks.

Joint phase mode also defers prefill when the time budget is not available. When only the unserved tail of the previous decode batch is old, it refreshes that tail rather than executing a whole batch of recently served requests again. This reduces the time needed to make room for prefill and preserves original request IDs through the existing runner/postprocess mapping.

A prefill step still gives way to a decode opportunity. A maximum consecutive decode count still guarantees waiting work an execution opportunity. If progress must be forced when no predicted candidate fits, this is counted explicitly. Non-preemptible kernels, CPU overhead and prediction error mean the service budget is not a hard latency guarantee.

## Switches and ablations

```python
llm = LLM(
    model_path,
    shared_attention_backend="auto",
    schedule_policy="cost",
    schedule_cost_profile="results/prefill/cost-profile.json",
    schedule_prefill_mode="model",
    schedule_phase_mode="predictive",
    schedule_service_budget_ms=70,
    schedule_prediction_margin=1.10,
    schedule_planning_margin_ms=2,
    schedule_queue_limit=96,
)
```

Both new switches default to the old behavior (`fixed`, `guarded`). `schedule_prefill_chunk` is the configured fallback/old-policy chunk, not the maximum predicted candidate size. `max_num_batched_tokens` remains the hard token budget. The 25 ms decode interval remains the urgency/group-deferral check; joint prefill admission uses its separate 70 ms soft completion budget. The benchmark acceptance standards remain TTFT <= 500 ms and request maximum ITL <= 75 ms.

Five matched variants are compared:

1. `baseline`: original prefill priority.
2. `fixed_v1`: previous cost scheduler with fixed 1024-token chunks.
3. `fixed_large`: same scheduler with fixed 4096-token chunks, showing what simple chunk enlargement achieves and costs.
4. `model_only`: measured candidate budget, previous guarded phase rule.
5. `joint`: measured candidate budget, predictive phase choice and targeted decode-tail refresh.

The comparison reuses the prior natural-language formatted synthetic documents, preset arrivals, output lengths, 96-entry admission cap, attention backend and SLO thresholds. Faster completion can reopen admission slots during overload, so accepted/rejected counts are reported per variant rather than assumed identical. All offered requests are accounted for, and accepted work is drained and checked for reference leaks.

## Validation and reproduction

- `tests/validate_prefill_model.py`: held-out shape error, batching, overdue-decode priority and unsupported-domain fallback.
- `tests/audit_prefill.py`: six separately measured shapes that never update the profile.
- `tests/validate_prefill_gpu.py`: identical per-request teacher-forced histories across differing chunk/batch/stage orders, supported and unsupported cached-context paths, and COW parent-tail immutability.
- Existing scheduler, COW and paged attention tests remain regression checks.

```bash
bash /root/autodl-tmp/shared-context/run_cow_env.sh python experiments/calibrate_prefill.py \
  --model /root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16 \
  --decode-profile results/scheduler/cost-profile.json \
  --output results/prefill/cost-profile.json
bash /root/autodl-tmp/shared-context/run_cow_env.sh python experiments/replay_prefill.py \
  --model /root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16 \
  --profile results/prefill/cost-profile.json --output results/prefill/replay.json
```

The old `pilot` results are diagnostics, not final performance evidence. Final conclusions use the frozen five-variant, five-scenario, five-repeat run and retain latency violations and rejection counts. This is finite-cohort online replay, not a production capacity or P99 guarantee.
