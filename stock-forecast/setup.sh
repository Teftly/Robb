#!/usr/bin/env bash
# Sets up the TimesFM 2.5 Python environment from scratch.
# Run from the repo root: bash stock-forecast/setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Creating virtual environment..."
uv venv .venv

echo "==> Cloning TimesFM..."
git clone https://github.com/google-research/timesfm.git timesfm

echo "==> Installing TimesFM + dependencies..."
source .venv/bin/activate
uv pip install -e "timesfm/[torch]"
uv pip install matplotlib yfinance

echo ""
echo "Done. Run forecasts with:"
echo "  source .venv/bin/activate"
echo "  python stock-forecast/stock_forecast.py"
