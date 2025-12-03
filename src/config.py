import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models_store"
LOG_DIR = ROOT_DIR / "logs"

for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LAST_FIVE_SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RANDOM_STATE = 42
TEST_SIZE = 0.2
