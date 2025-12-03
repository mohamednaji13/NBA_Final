#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "===================================="
echo "   NBA DATA PIPELINE (V3 + PROXY)"
echo "===================================="
echo ""

mkdir -p data/raw data/processed logs

echo "DATA COLLECTION STARTED: $(date)" > data/data_saving_indicator.txt

echo "[1/5] Fetching schedule (5 seasons)..."
python -m src.data.fetch_schedule
echo "✔ Schedule fetched"
echo ""

echo "[2/5] Fetching V3 boxscores via proxy (incremental save every 10 games)..."
python -m src.data.fetch_boxscores
echo "✔ Boxscores fetched"
echo ""

echo "[3/5] Building enriched dataset..."
python -m src.data.build_dataset
echo "✔ Dataset built"
echo ""

echo "[4/5] Training model..."
python -m src.models.train_model
echo "✔ Model trained"
echo ""

echo "[5/5] Evaluating model..."
python -m src.models.evaluate
echo "✔ Evaluation done"
echo ""

echo "DATA COLLECTION FINISHED: $(date)" >> data/data_saving_indicator.txt

echo "===================================="
echo "   All data saved under /app/data"
echo "   Check data/raw & data/processed"
echo "===================================="
