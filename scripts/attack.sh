#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python attack.py stego \
  --model_path "${MODEL_PATH:?set MODEL_PATH}" \
  --member_image_dir "${COVER_DIR:?set COVER_DIR}" \
  --member_token_file "${CAPTION_FILE:?set CAPTION_FILE}" \
  --stego_image_dir "${STEGO_DIR:?set STEGO_DIR}" \
  --clean_image_dir "${COVER_DIR}" \
  --output_dir "${OUTPUT_DIR:-./outputs/attack}" \
  --hyper_lambda "${HYPER_LAMBDA:-0.5}"
