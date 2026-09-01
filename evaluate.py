#!/usr/bin/env python3
"""Evaluation entry point."""
from _dispatch import dispatch

USAGE = """usage: python evaluate.py {classification,retrieval,quality} [args]
"""

if __name__ == "__main__":
    dispatch(
        {
            "classification": "src/eval/classification.py",
            "retrieval": "src/eval/retrieval.py",
            "quality": "src/eval/image_quality.py",
        },
        USAGE,
    )
