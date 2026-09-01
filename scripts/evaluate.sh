#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python evaluate.py quality \
  --clean_image_dir "${COVER_DIR:?set COVER_DIR}" \
  --stego_image_dir "${STEGO_DIR:?set STEGO_DIR}" \
  --output_dir "${OUTPUT_DIR:-./outputs/eval}"
