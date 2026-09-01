#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python train.py stego \
  --train_csv "${TRAIN_CSV:?set TRAIN_CSV}" \
  --clean_image_dir "${COVER_DIR:?set COVER_DIR}" \
  --stego_image_dir "${STEGO_DIR:?set STEGO_DIR}" \
  --pretrained_model_path "${PRETRAINED:?set PRETRAINED}" \
  --model "${MODEL:-ViT-B-16}" \
  --epochs "${EPOCHS:-10}" \
  --batch_size "${BATCH_SIZE:-32}"
