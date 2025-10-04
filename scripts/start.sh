#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-8000}"
DATA_DIR="${2:-}"

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel cmake
pip install -r requirements.txt

if [ -n "$DATA_DIR" ]; then
  export SMARTATTENDANCE_DATA_DIR="$DATA_DIR"
fi

python -m uvicorn web.main:app --host 0.0.0.0 --port "$PORT"