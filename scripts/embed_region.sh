#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python embed.py region \
  --foreground_dir "${FOREGROUND_DIR:?set FOREGROUND_DIR}" \
  --mask_dir "${MASK_DIR:?set MASK_DIR}" \
  --original_dir "${COVER_DIR:?set COVER_DIR}" \
  --caption_file "${CAPTION_FILE:?set CAPTION_FILE}" \
  --stego_foreground_dir "${STEGO_FOREGROUND_DIR:-./outputs/stego_foreground}" \
  --stego_output_dir "${STEGO_DIR:-./outputs/stego}" \
  --domain dct \
  --key "${KEY:-1234}"
