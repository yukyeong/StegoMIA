#!/usr/bin/env python3
"""Defense entry point."""
from _dispatch import dispatch

USAGE = """usage: python defense.py {finetune} [args]
"""

if __name__ == "__main__":
    dispatch(
        {
            "finetune": "src/defense/finetune.py",
        },
        USAGE,
    )
