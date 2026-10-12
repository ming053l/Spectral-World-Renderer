#!/usr/bin/env bash
# Usage: env/run.sh <gpu|-1> <script.py> [args...]   run a python script in the Abundance-GS environment on one GPU
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
if [ "$1" = "-1" ]; then export CUDA_VISIBLE_DEVICES=""; else export CUDA_VISIBLE_DEVICES="$1"; fi
shift
exec "$ABGS_PYTHON" "$@"
