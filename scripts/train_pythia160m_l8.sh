#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Each persistence coefficient is learned per feature; pass --num-persistence-groups N to share them.
python -m persistent_sae.train \
  --model EleutherAI/pythia-160m-deduped \
  --layer 8 \
  --dataset monology/pile-uncopyrighted \
  --dataset-split train \
  --ctx-len 256 \
  --seq-batch-size 8 \
  --n-ctxs 1000 \
  --refresh-batch-size 1024 \
  --dict-size 16384 \
  --k 20 \
  --persistence-init constant \
  --init-persistence 0.5 \
  --max-persistence 0.999 \
  --max-steps 244140 \
  --lr 3e-4 \
  --warmup-steps 1000 \
  --seed 0 \
  --save-root checkpoints \
  --checkpoint-start-step 1000 \
  --log-steps 50 \
  "$@"
