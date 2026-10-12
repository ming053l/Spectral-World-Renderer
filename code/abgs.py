#!/usr/bin/env python3
"""Abundance-GS command line (same as the `abgs` console script): python abgs.py <stage> [options]; see README.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True                      # never write .pyc into the code tree

from abundance_gs.entry import main  # noqa: E402

if __name__ == '__main__':
    main()
