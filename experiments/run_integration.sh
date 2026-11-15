#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/shared-context
export PATH="$ROOT/envs/main/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export TRITON_CACHE_DIR="$ROOT/.cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/inductor"
DEV="$ROOT/src/nano-vllm-shared"
export PYTHONPATH="$DEV"
python "$DEV/tests/validate_paged_shared.py" --output "$ROOT/results/integration/paged-correctness.json" > "$ROOT/logs/integration/paged-correctness.log" 2>&1
python "$DEV/tests/validate_engine_shared.py" --backend auto --model "$ROOT/models/Qwen3-0.6B-fp16" --output "$ROOT/results/integration/model-fp16-auto-graph-final.json" > "$ROOT/logs/integration/model-fp16-auto-graph-final.log" 2>&1
python "$DEV/tests/validate_engine_shared.py" --backend auto --eager --model "$ROOT/models/Qwen3-0.6B-fp16" --output "$ROOT/results/integration/model-fp16-auto-eager-fallback.json" > "$ROOT/logs/integration/model-fp16-auto-eager-fallback.log" 2>&1
python "$DEV/tests/validate_engine_shared.py" --backend auto --model "$ROOT/models/Qwen3-0.6B" --output "$ROOT/results/integration/model-bf16-auto-fallback.json" > "$ROOT/logs/integration/model-bf16-auto-fallback.log" 2>&1
for backend in original off auto force; do
  if [ "$backend" = original ]; then
    export PYTHONPATH="$ROOT/src/nano-vllm"
  else
    export PYTHONPATH="$DEV"
  fi
  python "$ROOT/experiments/integration/bench_integrated.py" --backend "$backend" \
    --model "$ROOT/models/Qwen3-0.6B-fp16" \
    --output "$ROOT/results/integration/serving-$backend-graph.json" \
    > "$ROOT/logs/integration/serving-$backend-graph.log" 2>&1
done
export PYTHONPATH="$DEV"
for backend in off auto; do
  python "$ROOT/experiments/integration/bench_integrated.py" --backend "$backend" --eager \
    --scenarios shared_long small_shared no_shared --model "$ROOT/models/Qwen3-0.6B-fp16" \
    --output "$ROOT/results/integration/serving-$backend-eager.json" \
    > "$ROOT/logs/integration/serving-$backend-eager-final.log" 2>&1
done
echo INTEGRATION_COMPARISON_COMPLETE
