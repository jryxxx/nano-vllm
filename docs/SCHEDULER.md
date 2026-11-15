# Fair and shared-prefix-aware scheduling

This phase extends the COW/shared-attention engine with independent policies. The previous checkout is preserved at `/root/autodl-tmp/shared-context/src/nano-vllm-pre-scheduler`. `baseline` remains the default; enabling scheduling does not silently change the attention backend.

## Policies

- `baseline`: upstream-style prefill priority and original decode queue behavior.
- `fcfs`: bounded prefill chunks, decode service-age checks, a maximum consecutive decode count while prefill work is waiting, oldest-arrival prefill service, and oldest-last-service decode rotation.
- `lpm`: the same fairness mechanisms, with cached-prefix priority among up to 16 non-overdue waiting candidates and shared physical-prefix group selection for decode. Group score is common complete-prefix length times group size. This is the project's bounded prefix-priority baseline, not a reproduction of another engine's entire LPM implementation.
- `cost`: the same stage fairness and waiting-prefix heuristic, with measured decode cost comparison. It compares fair mixed-batch tokens/ms to candidate-group tokens/ms, requires at least 5% estimated improvement, and refuses a group when its predicted duration would push deferred work across the configured decode service interval.

Group discovery examines at most `schedule_candidate_limit` ready requests and retains at most `schedule_max_groups` distinct keys. Keys use the first `shared_min_prefix` worth of complete physical pages, whose ownership is protected by the existing synchronous cache manager. Each group is verified again by the existing attention dispatcher. This is a bounded, single-level prefix grouping implementation; it does not search every nested prefix boundary.

One group executes per engine step. The selected sequence objects are passed to the runner in the chosen order, sampled in that order, then postprocessed against those same objects. Deferred requests remain runnable and completed requests retain their original IDs. There is no new multi-group fused kernel or row-reordering inside a captured graph. The existing ordinary/shared graph banks are reused for each selected batch.

## Fairness and admission

When prefill and decode both exist:

1. A prefill step is followed by a decode opportunity.
2. Decode requests exceeding the service interval get supplemental decode opportunities, including when the runnable queue is larger than one batch.
3. After the configured maximum consecutive decode steps, waiting prefill gets an opportunity.
4. Oldest waiting arrivals override prefix preference after the waiting threshold.

These are service-opportunity guards, not hard end-to-end deadlines. A CUDA kernel cannot be preempted mid-step; waiting through a complete prompt can span multiple chunks. Offered load above processing capacity still produces long queues and missed SLOs. A bounded active-request registry provides explicit backpressure through `RequestQueueFull`, including atomic bulk-fork rejection before KV references are acquired. Paused requests count toward the limit; saved snapshots do not count as runnable requests but retain KV ownership.

## Configuration

```python
llm = LLM(
    model_path,
    shared_attention_backend="auto",
    schedule_policy="cost",  # baseline / fcfs / lpm / cost
    schedule_cost_profile="results/scheduler/cost-profile.json",
    schedule_prefill_chunk=1024,
    schedule_decode_interval_ms=25,
    schedule_wait_ms=100,
    schedule_max_decode_steps=4,
    schedule_queue_limit=96,
)
```

The queue limit defaults to 0 (unlimited) for compatibility; a serving deployment should set a finite limit. Candidate limit defaults to 512; at most four groups are considered by default. Session COW APIs retain their single-GPU/single-thread scope.

The decode table records model, GPU, execution mode, batch size, context length, common physical-prefix availability, selected attention route and complete engine-step duration. It is calibrated on this RTX 4090/FP16 model and should be regenerated on another platform. Lookup requires the same route/common-prefix category and a nearby batch/context shape (log2 distance at most 0.6). Unknown configurations fall back to fair FCFS. Measured executions update neighboring entries with a 0.1 EWMA weight; online entries are capped at 128.

Prefill currently uses a fixed bounded token chunk. An initial ms/token adaptation shrank chunks below a useful size because it charged fixed execution overhead as per-token work; the resulting cold-request slowdown is retained in diagnostic artifacts. A calibrated affine/context-dependent Prefill time model remains future work. The current implementation does not promise that a 1024-token chunk fits a fixed number of milliseconds.

## Validation and replay

- `tests/validate_scheduler.py`: deterministic stage opportunities, decode rotation, measured group choice, overdue override and admission rejection.
- `tests/validate_scheduler_gpu.py`: per-request/per-token comparison across different scheduling orders and batch shapes, plus mixed chunked prefill/decode and bulk-fork rejection with unchanged references.
- Existing COW, paged attention and engine integration tests remain regression checks.
- `experiments/calibrate_scheduler.py`: offline decode lookup calibration.
- `experiments/replay_scheduler.py`: preset open-loop arrivals, fixed-format natural-language synthetic documents, real sampling, fixed output lengths, explicit rejection and complete cohort accounting.

Replay counts latency from the intended arrival time, including time waiting outside the engine while a step executes. Hot sessions are materialized before timed arrivals; actual client fork requests are created only at their scheduled times. Cold prompts are not prefetched. Goodput counts completed requests meeting both TTFT <= 500 ms and maximum per-request ITL <= 75 ms, divided by the elapsed cohort drain time. Rejected, unfinished or late requests never count as good. It is a finite-cohort replay, not a sustained production-capacity or P99 certification.

The end-to-end report must include both hot/cold cohorts, rejected requests, raw throughput and goodput: stronger decode service can improve hot latency while reducing total throughput and increasing cold TTFT. All four policies use the same queue cap, model, attention setting and workload.

## Reproduction

```bash
bash /root/autodl-tmp/shared-context/run_cow_env.sh python experiments/calibrate_scheduler.py \
  --model /root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16 \
  --output results/scheduler/cost-profile.json
bash /root/autodl-tmp/shared-context/run_cow_env.sh python experiments/replay_scheduler.py \
  --model /root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16 \
  --profile results/scheduler/cost-profile.json --output results/scheduler/replay.json
```
