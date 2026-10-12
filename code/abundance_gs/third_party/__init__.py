"""Vendored third-party code (origins, versions and licences: THIRD_PARTY.md).

`gsplat` keeps its own absolute imports, so this directory is put first on sys.path by `use()`; the CUDA libraries
are built and loaded by abundance_gs/ext.py.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def use():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
