#!/usr/bin/env python3
"""Membership-inference entry point."""
from _dispatch import dispatch

USAGE = """usage: python attack.py {stego,cosine} [args]

stego      membership inference via frequency-domain image steganography
cosine     cosine-similarity membership inference
"""

if __name__ == "__main__":
    dispatch(
        {
            "stego": "src/attack/mia_stego.py",
            "cosine": "src/attack/mia_cosine.py",
        },
        USAGE,
    )
