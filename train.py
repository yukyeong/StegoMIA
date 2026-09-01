#!/usr/bin/env python3
"""Training entry point."""
from _dispatch import dispatch

USAGE = """usage: python train.py {prepare,stego,full} [args]

prepare    build mixed clean/stego CSV files
stego      train on a sampled stego ratio
full       train on the full stego set
"""

if __name__ == "__main__":
    dispatch(
        {
            "prepare": "src/train/prepare_data.py",
            "stego": "src/train/train_stego.py",
            "full": "src/train/train_stego_full.py",
        },
        USAGE,
    )
