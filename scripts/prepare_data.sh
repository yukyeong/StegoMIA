#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python train.py prepare \
  --stego_dir "${STEGO_DIR:?set STEGO_DIR}" \
  --clean_dir "${COVER_DIR:?set COVER_DIR}" \
  --caption_file "${CAPTION_FILE:?set CAPTION_FILE}" \
  --output_dir "${OUTPUT_DIR:-./outputs/data_configs}" \
  --ratios ${RATIOS:-0.01 0.02 0.05}
