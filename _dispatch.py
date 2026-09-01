#!/usr/bin/env python3
"""Dispatch helper used by the top-level entry points."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def dispatch(mapping: dict[str, str], usage: str) -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(usage)
        raise SystemExit(0)
    command = sys.argv.pop(1)
    if command not in mapping:
        raise SystemExit(f"Unknown command {command!r}.\n{usage}")
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / mapping[command]), run_name="__main__")
