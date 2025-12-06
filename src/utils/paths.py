# src/utils/paths.py
from pathlib import Path

# Root of the project ( /app inside Docker )
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Schedule + Boxscores files
SCHEDULE_CSV = RAW_DIR / "games_schedule.csv"
BOXSCORES_CSV = RAW_DIR / "boxscores.csv"
BOXSCORES_PARTIAL_CSV = RAW_DIR / "boxscores_partial.csv"
BOXSCORES_PLAYERS_CSV = RAW_DIR / "boxscores_players.csv"
BOXSCORES_PLAYERS_PARTIAL_CSV = RAW_DIR / "boxscores_players_partial.csv"

# Ensure directories exist
for p in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
