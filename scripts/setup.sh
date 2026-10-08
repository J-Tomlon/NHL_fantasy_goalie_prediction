#!/usr/bin/env bash
# One-time setup on macOS / Linux:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "Creating .venv with $($PY --version)"
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then try:             python -m goalie_predictor predict"
