#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Embed full captions into entire clean images (full-view steganography).

Unlike embed_region.py, this writes frequency-domain payloads
directly into the whole image without foreground/mask compositing.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm

from src.stego.frequency import embed_frequency, embed_frequency_dct  # noqa: E402

from src.stego.embed_region import (  # noqa: E402
    get_caption_for_image,
    index_images,
    load_captions,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}


def embed_full_image(
    image_path: str,
    output_path: str,
    caption: str,
    domain: str,
    key: int,
    block_size: int,
    freq: Tuple[int, int],
    min_magnitude: float,
) -> bool:
    try:
        cover = Image.open(image_path).convert("RGB")
        payload = caption.encode("utf-8")
        if domain == "dct":
            stego = embed_frequency_dct(
                cover,
                payload,
                key=key,
                block_size=block_size,
                freq=freq,
                min_magnitude=min_magnitude,
            )
        else:
            stego = embed_frequency(
                cover,
                payload,
                key=key,
                block_size=block_size,
                freq=freq,
                min_magnitude=min_magnitude,
            )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        if ext in {".jpg", ".jpeg"}:
            stego.save(output_path, format="JPEG", quality=95)
        elif ext == ".png":
            stego.save(output_path, format="PNG")
        else:
            stego.save(output_path, format="JPEG", quality=95)
        return True
    except Exception as exc:
        print(f"Error embedding {image_path}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-view caption steganography")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory of clean images")
    parser.add_argument("--caption_file", type=str, required=True, help="Caption token/json file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output stego image directory")
    parser.add_argument("--domain", type=str, default="fft", choices=["fft", "dct"])
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--freq", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--min_magnitude", type=float, default=2.0)
    parser.add_argument("--sample_fraction", type=float, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    print("Loading captions...")
    captions = load_captions(args.caption_file)
    print(f"Loaded {len(captions)} captions")

    originals = index_images(args.clean_dir)
    if not originals:
        raise SystemExit(f"No images found in {args.clean_dir}")

    bases = sorted(originals.keys())
    if args.sample_fraction is not None:
        rng = random.Random(args.seed)
        rng.shuffle(bases)
        bases = bases[: max(1, int(len(bases) * args.sample_fraction))]
    if args.max_images is not None:
        bases = bases[: args.max_images]

    print(f"Processing {len(bases)} images -> {args.output_dir}")
    ok = skip = fail = 0
    for base in tqdm(bases, desc="Full-view stego"):
        src_path = originals[base]
        image_name = src_path.name
        dst_path = os.path.join(args.output_dir, image_name)
        if args.skip_existing and os.path.exists(dst_path):
            skip += 1
            continue
        caption = get_caption_for_image(image_name, captions)
        if not caption:
            fail += 1
            continue
        if embed_full_image(
            str(src_path),
            dst_path,
            caption,
            args.domain,
            args.key,
            args.block_size,
            tuple(args.freq),
            args.min_magnitude,
        ):
            ok += 1
        else:
            fail += 1

    print(f"Done: success={ok}, skipped={skip}, failed={fail}")


if __name__ == "__main__":
    main()
