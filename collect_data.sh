#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH. Please install Python 3 and re-run." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Ensure project root is on PYTHONPATH so `python3 -m src...` works regardless of caller's CWD.
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo ""
echo "===================================="
echo "   NBA DATA PIPELINE (V3 + PROXY)"
echo "===================================="
echo ""

mkdir -p data/raw data/processed logs

# Clear prior CSV outputs to ensure a clean rebuild
rm -f data/raw/boxscores*.csv data/raw/games_schedule*.csv data/processed/*.csv

echo "DATA COLLECTION STARTED: $(date)" > data/data_saving_indicator.txt

echo "[1/5] Fetching schedule (5 seasons)..."
python3 -m src.data.fetch_schedule
echo "✔ Schedule fetched"
echo ""

echo "[2/5] Fetching V3 boxscores via proxy (incremental save every 10 games)..."
python3 -m src.data.fetch_boxscores
echo "✔ Boxscores fetched"
echo ""

echo "[3/5] Building enriched dataset..."
python3 -m src.data.build_dataset
echo "✔ Dataset built"
echo ""

echo "[4/5] Training model..."
python3 -m src.models.train_model
echo "✔ Model trained"
echo ""

echo "[5/5] Evaluating model..."
python3 -m src.models.evaluate
echo "✔ Evaluation done"
echo ""

echo "DATA COLLECTION FINISHED: $(date)" >> data/data_saving_indicator.txt

echo "===================================="
echo "   All data saved under /app/data"
echo "   Check data/raw & data/processed"
echo "===================================="
