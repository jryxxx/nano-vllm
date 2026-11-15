#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/shared-context
cd "$BASE/src/nano-vllm-shared"
ENV="$BASE/run_cow_env.sh"
MODEL="$BASE/models/Qwen3-0.6B-fp16"
OUT=results/finalization
run() { local name="$1"; shift; bash "$ENV" python "$@" > "$OUT/$name.log" 2>&1; printf '%s completed\n' "$name"; }
run paged-final tests/validate_paged_shared.py --output "$OUT/paged-final.json"
run scheduler-final tests/validate_scheduler_gpu.py --model "$MODEL" --profile "$OUT/cost-profile.json" --output "$OUT/scheduler-final.json"
run combined-validation tests/validate_combined.py --model "$MODEL" --output "$OUT/combined-validation.json"
