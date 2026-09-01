#!/usr/bin/env python3
"""Embedding entry point."""
from _dispatch import dispatch

USAGE = """usage: python embed.py {region,full,highband,codec} [args]

region     embed captions into a foreground region and composite onto the cover
full       embed captions into the full image
highband   embed captions into high-frequency DCT bins
codec      low-level encode/decode for a single image
"""

if __name__ == "__main__":
    dispatch(
        {
            "region": "src/stego/embed_region.py",
            "full": "src/stego/embed_full.py",
            "highband": "src/stego/embed_highband.py",
            "codec": "src/stego/frequency.py",
        },
        USAGE,
    )
