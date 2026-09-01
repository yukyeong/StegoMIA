#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Embed each image's first matching caption in high-frequency DCT bins.

The previous implementation added one static red-team trigger to every image;
it did not consume a caption file and therefore was not caption steganography.
This implementation uses the project's length-prefixed, keyed DCT codec and
verifies the saved image by decoding the exact UTF-8 payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stego.frequency import (  # noqa: E402
    embed_frequency_dct,
    extract_frequency_dct,
)
from src.stego.embed_region import (  # noqa: E402
    get_caption_for_image,
    load_captions,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# All coordinates are in the high-frequency region of an 8x8 DCT block.
# (7, 7) is used whenever possible; fallbacks handle rare clipping/JPEG cases.
DEFAULT_HIGH_FREQS: Tuple[Tuple[int, int], ...] = (
    (7, 7), (7, 6), (6, 7), (6, 6), (7, 5), (5, 7),
    (6, 5), (5, 6), (5, 5), (7, 4), (4, 7),
) + tuple(
    (row, col)
    for row in range(2, 8)
    for col in range(2, 8)
    if row + col >= 9
    and (row, col) not in {
        (7, 7), (7, 6), (6, 7), (6, 6), (7, 5), (5, 7),
        (6, 5), (5, 6), (5, 5), (7, 4), (4, 7),
    }
)


def list_images(directory: Path) -> List[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def save_image(image: Image.Image, path: str, jpeg_quality: int) -> None:
    ext = Path(path).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=jpeg_quality)
    elif ext == ".png":
        image.save(path, format="PNG")
    else:
        raise ValueError(f"Unsupported output extension: {ext}")


def decode_matches(
    image: Image.Image,
    payload: bytes,
    key: int,
    block_size: int,
    freq: Tuple[int, int],
) -> bool:
    try:
        recovered = extract_frequency_dct(
            image, key=key, block_size=block_size, freq=freq,
        )
        return recovered == payload
    except Exception:
        return False


def process_one(
    src_path: str,
    dst_path: str,
    caption: str,
    key: int,
    block_size: int,
    frequencies: Sequence[Tuple[int, int]],
    min_magnitude: float,
    jpeg_quality: int,
    max_attempts: int,
) -> Dict[str, object]:
    payload = caption.encode("utf-8")
    dst = Path(dst_path)
    tmp_path = str(dst.with_name(f".{dst.stem}.tmp-{os.getpid()}{dst.suffix}"))
    try:
        cover = Image.open(src_path).convert("RGB")
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

        for freq in frequencies:
            # Restart from the source when changing coordinates so fallback
            # trials do not accumulate payloads in multiple frequency bins.
            current = cover.copy()
            for attempt in range(1, max_attempts + 1):
                stego = embed_frequency_dct(
                    current,
                    payload,
                    key=key,
                    block_size=block_size,
                    freq=freq,
                    min_magnitude=min_magnitude,
                    device="cpu",
                )
                save_image(stego, tmp_path, jpeg_quality)
                with Image.open(tmp_path) as saved:
                    current = saved.convert("RGB")
                    current.load()
                if decode_matches(current, payload, key, block_size, freq):
                    os.replace(tmp_path, dst_path)
                    return {
                        "status": "ok",
                        "image": Path(src_path).name,
                        "caption": caption,
                        "caption_bytes": len(payload),
                        "caption_sha256": hashlib.sha256(payload).hexdigest(),
                        "freq": list(freq),
                        "attempts": attempt,
                        "verified_exact": True,
                    }

        return {
            "status": "fail",
            "image": Path(src_path).name,
            "error": "saved image did not decode to the exact caption",
        }
    except Exception as exc:
        return {
            "status": "fail",
            "image": Path(src_path).name,
            "error": repr(exc),
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _process_job(job: Tuple[object, ...]) -> Dict[str, object]:
    return process_one(*job)


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run_dataset(
    name: str,
    source_dir: Path,
    caption_file: Path,
    output_dir: Path,
    manifest_dir: Path,
    key: int,
    block_size: int,
    frequencies: Sequence[Tuple[int, int]],
    min_magnitude: float,
    jpeg_quality: int,
    max_attempts: int,
    workers: int,
) -> Dict[str, object]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")
    if not caption_file.is_file():
        raise FileNotFoundError(f"Missing caption file: {caption_file}")

    captions = load_captions(str(caption_file))
    images = list_images(source_dir)
    if not images:
        raise RuntimeError(f"No images found in {source_dir}")

    matched: List[Tuple[Path, str]] = []
    missing: List[str] = []
    for image_path in images:
        caption = get_caption_for_image(image_path.name, captions)
        if caption:
            matched.append((image_path, caption))
        else:
            missing.append(image_path.name)
    if missing:
        examples = ", ".join(missing[:5])
        raise RuntimeError(
            f"{name}: {len(missing)} images have no matching caption; examples: {examples}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Processing {len(matched)} caption-matched images: "
        f"{source_dir} -> {output_dir}"
    )
    jobs = [
        (
            str(src), str(output_dir / src.name), caption, key, block_size,
            tuple(frequencies), min_magnitude, jpeg_quality, max_attempts,
        )
        for src, caption in matched
    ]

    records: List[Dict[str, object]] = []
    if workers <= 1:
        iterator = map(_process_job, jobs)
        for record in tqdm(iterator, total=len(jobs), desc=output_dir.name):
            records.append(record)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            iterator = pool.map(_process_job, jobs, chunksize=8)
            for record in tqdm(iterator, total=len(jobs), desc=output_dir.name):
                records.append(record)

    failures = [row for row in records if row["status"] != "ok"]
    verified = [row for row in records if row["status"] == "ok"]
    if failures:
        print(f"{name}: failed={len(failures)}; first failures: {failures[:5]}")

    records.sort(key=lambda row: str(row["image"]))
    manifest_path = manifest_dir / f"{output_dir.name}.caption_manifest.jsonl"
    write_jsonl_atomic(manifest_path, records)

    freq_counts = Counter(tuple(row["freq"]) for row in verified)
    attempt_counts = Counter(int(row["attempts"]) for row in verified)
    summary: Dict[str, object] = {
        "dataset": name,
        "source_dir": str(source_dir),
        "caption_file": str(caption_file),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "codec": "length-prefixed keyed block-DCT sign encoding",
        "key": key,
        "block_size": block_size,
        "primary_freq": list(frequencies[0]),
        "fallback_freqs": [list(freq) for freq in frequencies[1:]],
        "min_magnitude": min_magnitude,
        "jpeg_quality": jpeg_quality,
        "max_attempts_per_frequency": max_attempts,
        "source_images": len(images),
        "caption_matches": len(matched),
        "verified_exact": len(verified),
        "failed": len(failures),
        "frequency_counts": {
            f"{r},{c}": n for (r, c), n in sorted(freq_counts.items())
        },
        "attempt_counts": {str(k): v for k, v in sorted(attempt_counts.items())},
    }
    write_json_atomic(
        manifest_dir / f"{output_dir.name}.summary.json", summary,
    )
    print(
        f"Done {output_dir.name}: verified_exact={len(verified)}, "
        f"failed={len(failures)}, manifest={manifest_path}"
    )
    if failures:
        raise RuntimeError(
            f"{name}: {len(failures)} images failed exact caption verification"
        )
    return summary


def default_jobs(dataset_root: Path):
    return [
        (
            "Flickr8k",
            dataset_root / "Flickr8k_Captains" / "stego_Flicker8k_Dataset",
            dataset_root / "Flickr8k_Captains" / "Flickr8k.token.txt",
            dataset_root / "Flickr8k_Captains" / "High_frequency_Flicker8k_Dataset",
        ),
        (
            "MSCOCO2017",
            dataset_root / "MSCOCO2017" / "subject_stego_train_Dataset",
            dataset_root / "MSCOCO2017" / "captions_train2017.json",
            dataset_root / "MSCOCO2017" / "High_frequency_MSCOCO2017_Dataset",
        ),
        (
            "ImageNet-1k",
            dataset_root / "Imagenet-1k" / "subject_stego_train_Dataset",
            dataset_root / "Imagenet-1k" / "captions_train.json",
            dataset_root / "Imagenet-1k" / "High_frequency_Imagenet-1k_Dataset",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verified high-frequency caption embedding"
    )
    parser.add_argument("--source_dir", type=Path)
    parser.add_argument("--caption_file", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--dataset_name", default="custom")
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--freq", type=int, nargs=2, default=[7, 7])
    parser.add_argument("--min_magnitude", type=float, default=40.0)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--max_attempts", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--manifest_dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "high_frequency" / "manifests",
    )
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    primary = tuple(args.freq)
    if any(v < 0 or v >= args.block_size for v in primary):
        raise SystemExit("--freq coordinates must be inside the DCT block")
    frequencies = (primary,) + tuple(
        freq for freq in DEFAULT_HIGH_FREQS if freq != primary
    )

    if args.all or not any((args.source_dir, args.caption_file, args.output_dir)):
        jobs = default_jobs(PROJECT_ROOT / "datasets")
    else:
        if not all((args.source_dir, args.caption_file, args.output_dir)):
            raise SystemExit(
                "Provide --source_dir, --caption_file and --output_dir together"
            )
        jobs = [(args.dataset_name, args.source_dir, args.caption_file, args.output_dir)]

    summaries = []
    for name, source_dir, caption_file, output_dir in jobs:
        summaries.append(
            run_dataset(
                name, source_dir, caption_file, output_dir,
                args.manifest_dir, args.key, args.block_size, frequencies,
                args.min_magnitude, args.jpeg_quality, args.max_attempts,
                args.workers,
            )
        )
    write_json_atomic(args.manifest_dir / "all_datasets.summary.json", summaries)


if __name__ == "__main__":
    main()
