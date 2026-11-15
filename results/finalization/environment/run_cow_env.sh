#!/bin/bash
set -euo pipefail
export PATH=/root/autodl-tmp/shared-context/envs/main/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/shared-context/src/nano-vllm-shared
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export TRITON_CACHE_DIR=/root/autodl-tmp/shared-context/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/shared-context/.cache/inductor
cd /root/autodl-tmp/shared-context/src/nano-vllm-shared
exec "$@"
