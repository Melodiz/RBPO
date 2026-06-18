#!/usr/bin/env bash
# Reproduce every paper figure from the data tracked in results/.
# No GPU, no k2, no Google Drive data required.
#
# Usage:  bash scripts/reproduce_figures.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Regenerating figures into paper/figures/ ..."
python scripts/generate_figures.py

echo
echo "Done. Rebuild the preprint PDF with:"
echo "    cd paper && ./build.sh"
