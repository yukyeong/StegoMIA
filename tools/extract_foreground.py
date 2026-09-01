#!/usr/bin/env python3
"""Extract foreground regions using user-provided binary masks.

This tool does not bundle a segmenter. Provide a directory of binary masks
with the same filename stems as the cover images. Non-zero mask pixels are
treated as foreground.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def index_images(directory: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            index[path.stem] = path
    return index


def extract_one(cover_path: Path, mask_path: Path, fg_out: Path, mask_out: Path) -> None:
    cover = Image.open(cover_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != cover.size:
        mask = mask.resize(cover.size, Image.NEAREST)
    mask_arr = np.array(mask)
    binary = (mask_arr > 127).astype(np.uint8) * 255
    binary_img = Image.fromarray(binary, mode="L")
    fg = Image.new("RGB", cover.size, (255, 255, 255))
    fg.paste(cover, mask=binary_img)
    fg_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    fg.save(fg_out)
    binary_img.save(mask_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract foregrounds from cover images using masks")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of cover images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory of binary masks")
    parser.add_argument("--foreground_dir", type=str, required=True, help="Output directory for foreground images")
    parser.add_argument("--mask_out_dir", type=str, default=None, help="Optional output directory for normalized masks")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    fg_dir = Path(args.foreground_dir)
    mask_out_dir = Path(args.mask_out_dir) if args.mask_out_dir else fg_dir.parent / "masks"

    covers = index_images(image_dir)
    masks = index_images(mask_dir)
    stems = sorted(set(covers) & set(masks))
    if not stems:
        raise SystemExit("No matching cover/mask filename stems were found.")

    for stem in tqdm(stems, desc="extract_foreground"):
        extract_one(
            covers[stem],
            masks[stem],
            fg_dir / f"{stem}.png",
            mask_out_dir / f"{stem}.png",
        )
    print(f"Wrote {len(stems)} foreground images to {fg_dir}")


if __name__ == "__main__":
    main()
