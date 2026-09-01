#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python defense.py finetune \
  --name defense_finetune \
  --train_data "${COVER_DIR:?set COVER_DIR}" \
  --train_token_file "${CAPTION_FILE:?set CAPTION_FILE}" \
  --checkpoint "${CHECKPOINT:?set CHECKPOINT}" \
  --logs ./outputs/logs \
  --save_final
