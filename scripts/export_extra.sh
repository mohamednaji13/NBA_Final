#!/usr/bin/env bash
set -euo pipefail

# Exports season/game/plus_minus rows into data/processed/extra.csv.
# Requires data/raw/games_schedule.csv to exist.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHED="$ROOT/data/raw/games_schedule.csv"
OUT="$ROOT/data/raw/extra.csv"

if [[ ! -f "$SCHED" ]]; then
  echo "Missing schedule: $SCHED" >&2
  exit 1
fi

python3 - <<'PY'
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sched_path = root / "data" / "raw" / "games_schedule.csv"
out_path = root / "data" / "raw" / "extra.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(sched_path, dtype={"GAME_ID": str, "TEAM_ID": str})

season_col = None
for col in ("SEASON_ID", "SEASON_LABEL", "SEASON"):
    if col in df.columns:
        season_col = col
        break

required = ["GAME_ID"]
if season_col:
    required.append(season_col)
if "PLUS_MINUS" in df.columns:
    required.append("PLUS_MINUS")
else:
    df["PLUS_MINUS"] = 0
    required.append("PLUS_MINUS")

missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Required columns missing from schedule: {missing}")

extra = df[required].drop_duplicates(subset=["GAME_ID"])
extra.to_csv(out_path, index=False)
print(f"Wrote {len(extra)} rows to {out_path}")
PY
