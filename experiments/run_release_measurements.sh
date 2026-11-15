#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/shared-context
cd "$BASE/src/nano-vllm-shared"
ENV="$BASE/run_cow_env.sh"
MODEL="$BASE/models/Qwen3-0.6B-fp16"
OUT=results/finalization
run_main() {
  local name="$1"
  shift
  bash "$ENV" python "$@" > "$OUT/$name.log" 2>&1
  printf '%s completed\n' "$name"
}
run_main release-core experiments/run_final_suite.py --group core
run_main release-sessions experiments/run_final_suite.py --group sessions
run_main release-cache experiments/run_final_suite.py --group cache
run_main release-tier experiments/run_final_suite.py --group tier
run_main release-openloop experiments/replay_prefill.py --model "$MODEL" --profile "$OUT/cost-profile.json" --output "$OUT/openloop-final.json" --repeats 3
run_main release-hf-reference tests/validate_precision_reference.py
run_main release-shared-fp16 tests/validate_engine_shared.py --model "$MODEL" --backend auto --output "$OUT/shared-fp16-final.json"
run_main release-shared-bf16 tests/validate_engine_shared.py --model "$BASE/models/Qwen3-0.6B" --backend auto --output "$OUT/shared-bf16-final.json"
run_main release-shared-4b tests/validate_engine_shared.py --model "$BASE/models/Qwen3-4B-fp16" --backend force --output "$OUT/shared-4b-final.json"
run_main reference-nano-off experiments/bench_external_engine.py --backend nano-off --model "$MODEL" --output "$OUT/reference-nano-off.json"
run_main reference-nano-auto experiments/bench_external_engine.py --backend nano-auto --model "$MODEL" --output "$OUT/reference-nano-auto.json"
run_main reference-4b-off experiments/bench_external_engine.py --backend nano-off --model "$BASE/models/Qwen3-4B-fp16" --gpu-util .85 --output "$OUT/reference-4b-off.json"
run_main reference-4b-force experiments/bench_external_engine.py --backend nano-force --model "$BASE/models/Qwen3-4B-fp16" --gpu-util .85 --output "$OUT/reference-4b-force.json"
VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
 "$BASE/envs/vllm-reference/bin/python" experiments/bench_external_engine.py --backend vllm --model "$MODEL" --output "$OUT/reference-vllm.json" > "$OUT/reference-vllm.log" 2>&1
printf 'reference-vllm completed\n'
