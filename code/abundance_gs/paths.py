"""Locations outside the package: the build directory of the CUDA extensions (data paths live in configs/).

The interpreter and CUDA toolkit come from env/env.sh (environment built by env/setup.sh).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                    # the folder holding abgs.py

# Writable directory of the compiled CUDA extensions (ext.py); one subdirectory per extension.
BUILD_DIR = Path(os.environ.get('ABGS_BUILD_DIR', '~/.cache/abundance_gs/build')).expanduser()
