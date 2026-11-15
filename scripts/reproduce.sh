#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ $# -lt 1 ]]; then echo 'Usage: bash scripts/reproduce.sh /absolute/model/path [calibrate|verify|bench|all]' >&2; exit 2; fi
export NANO_MODEL="$(cd "$1" && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PY="${PYTHON:-python}"
OUT=results/finalization
MODE="${2:-verify}"
mkdir -p "$OUT"
calibrate() {
 "$PY" experiments/calibrate_scheduler.py --model "$NANO_MODEL" --output "$OUT/decode-profile.json"
 "$PY" experiments/calibrate_prefill.py --model "$NANO_MODEL" --decode-profile "$OUT/decode-profile.json" --output "$OUT/cost-profile.json"
}
verify() {
 "$PY" experiments/run_final_suite.py --group core
 "$PY" tests/validate_hybrid_norm.py
 "$PY" tests/validate_engine_shared.py --model "$NANO_MODEL" --backend auto --output "$OUT/shared-fp16-final.json"
 "$PY" tests/validate_paged_shared.py --output "$OUT/paged-final.json"
 "$PY" tests/validate_scheduler_gpu.py --model "$NANO_MODEL" --profile "$OUT/cost-profile.json" --output "$OUT/scheduler-final.json"
 "$PY" tests/validate_combined.py --model "$NANO_MODEL" --output "$OUT/combined-validation.json"
 "$PY" tests/validate_greedy.py --model "$NANO_MODEL" --output "$OUT/greedy.json"
}
bench() {
 upstream="${NANO_UPSTREAM:-$ROOT/vendor/nano-vllm-upstream}"
 test "$(git -C "$upstream" rev-parse HEAD)" = bb823b3e06983d71485a8e1f23715ebd87d98ef8
 for group in sessions cache tier; do "$PY" experiments/run_final_suite.py --group "$group"; done
 for variant in Full-no-M Full-no-K Full-no-S; do
  "$PY" experiments/replay_sessions.py --model "$NANO_MODEL" --variant "$variant" --output "$OUT/sessions-$variant.json"
 done
 "$PY" experiments/replay_prefill.py --model "$NANO_MODEL" --profile "$OUT/cost-profile.json" --output "$OUT/openloop-final.json" --repeats 3
 "$PY" experiments/bench_combined.py --model "$NANO_MODEL" --output "$OUT/combined-final.json" --repeats 3
}
case "$MODE" in
 calibrate) calibrate;;
 verify) verify;;
 bench) bench;;
 all) calibrate; verify; bench;;
 *) echo "Unknown mode: $MODE" >&2; exit 2;;
esac
