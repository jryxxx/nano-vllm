#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/shared-context
cd "$BASE/src/nano-vllm-shared"
ENV="$BASE/run_cow_env.sh"
MODEL="$BASE/models/Qwen3-0.6B-fp16"
OUT=results/finalization
run() { local name="$1"; shift; bash "$ENV" python "$@" > "$OUT/$name.log" 2>&1; printf '%s completed\n' "$name"; }
run fresh-audit tests/audit_prefill.py --model "$MODEL" --profile "$OUT/cost-profile.json" --output "$OUT/fresh-audit.json"
run greedy tests/validate_greedy.py --model "$MODEL" --output "$OUT/greedy.json"
for variant in Full-no-M Full-no-K Full-no-S; do
 run "sessions-$variant" experiments/replay_sessions.py --model "$MODEL" --variant "$variant" --output "$OUT/sessions-$variant.json"
done
run combined-final experiments/bench_combined.py --model "$MODEL" --output "$OUT/combined-final.json" --repeats 3
