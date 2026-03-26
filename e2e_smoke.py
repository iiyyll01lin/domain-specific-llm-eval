#!/usr/bin/env python3
from __future__ import annotations

"""Compatibility wrapper for the in-process smoke test."""

from pathlib import Path
import runpy


if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent / 'scripts' / 'e2e_smoke.py'), run_name='__main__')