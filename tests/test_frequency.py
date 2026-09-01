from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stego.frequency import embed_frequency_dct, extract_frequency_dct  # noqa: E402


def test_dct_roundtrip_recovers_payload() -> None:
    cover = Image.new("RGB", (128, 128), color=(40, 80, 120))
    payload = b"stegomia-roundtrip"
    stego = embed_frequency_dct(
        cover,
        payload,
        key=1234,
        block_size=8,
        freq=(3, 3),
        min_magnitude=4.0,
        device="cpu",
    )
    recovered = extract_frequency_dct(stego, key=1234, block_size=8, freq=(3, 3))
    assert recovered == payload
