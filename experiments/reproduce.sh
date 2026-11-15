#!/usr/bin/env bash
# Reproduction on the recorded single-GPU server layout. Full overwrites result files.
set -euo pipefail
BASE=/root/autodl-tmp/shared-context
cd "$BASE/src/nano-vllm-shared"
MODE="${1:-verify}"
MODEL="$BASE/models/Qwen3-0.6B-fp16"
OUT=results/finalization
ENV="$BASE/run_cow_env.sh"
mkdir -p "$OUT"
case "$MODE" in
 verify)
  bash "$ENV" python experiments/run_final_suite.py --group core
  bash "$ENV" python tests/validate_hybrid_norm.py
  bash "$ENV" python tests/validate_greedy.py --model "$MODEL" --output "$OUT/greedy.json"
  bash experiments/run_final_checks.sh
  ;;
 full)
  test -f "$BASE/models/Qwen3-4B-fp16/config.json"
  test -x "$BASE/envs/vllm-reference/bin/python"
  test -d "$BASE/envs/flashinfer-reference/flashinfer"
  bash "$ENV" python experiments/calibrate_scheduler.py --model "$MODEL" --output "$OUT/decode-profile.json" > "$OUT/recalibrate-decode.log" 2>&1
  bash "$ENV" python experiments/calibrate_prefill.py --model "$MODEL" --decode-profile "$OUT/decode-profile.json" --output "$OUT/cost-profile.json" > "$OUT/recalibrate-prefill.log" 2>&1
  bash "$ENV" python tests/validate_hybrid_norm.py
  bash experiments/run_release_measurements.sh
  bash experiments/run_supplement.sh
  bash experiments/run_final_checks.sh
  bash "$ENV" python experiments/run_final_suite.py --group reference
  /root/miniconda3/bin/python experiments/summarize_release.py
  /root/miniconda3/bin/python experiments/write_release_report.py
  ;;
 summarize)
  /root/miniconda3/bin/python experiments/summarize_release.py
  /root/miniconda3/bin/python experiments/write_release_report.py
  ;;
 *) printf 'Usage: bash experiments/reproduce.sh [verify|full|summarize]\n' >&2; exit 2;;
esac
