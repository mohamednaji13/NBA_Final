"""
Daily updater that pulls:
- Yesterday's box scores (game + player level) from the NBA decodo scoreboard API.
- Today's active rosters for scheduled games.
- Appends the new games (0-25 per day) into data/processed/data.csv with season-average
  fill for any missing columns, then re-runs the training pipeline.

Run: python3 execution/daily_decodo_update.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
from nba_api.stats.endpoints import boxscoretraditionalv3, commonteamroster

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from merge import run_pipeline  # noqa: E402
from create_model.split_dataset import split_dataset  # noqa: E402
from create_model.train_test_val import main as train_models  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

BASE_PROCESSED = PROCESSED_DIR / "data.csv"
DAILY_PROCESSED = PROCESSED_DIR / "data_daily_temp.csv"
DAILY_ALIGNED = PROCESSED_DIR / "data_daily_aligned.csv"

TMP_BOX = RAW_DIR / "boxscores_yesterday.csv"
TMP_PLAYERS = RAW_DIR / "boxscores_players_yesterday.csv"
ACTIVE_PLAYERS_PATH = RAW_DIR / "active_players_today.csv"

SCOREBOARD_URL = "https://stats.nba.com/stats/scoreboardv3"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}


def season_id_from_year(year: int) -> int:
    """Map a season start year to NBA SEASON_ID format (e.g., 2024 -> 22024)."""
    return 22000 + (year % 10000)


def season_label_from_year(year: int) -> str:
    """Return label like '2024-25' from season start year."""
    return f"{year}-{str(year + 1)[-2:]}"


def parse_minutes(val) -> float | None:
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val)
    if ":" in text:
        try:
            mins, secs = text.split(":")
            return float(mins) + float(secs) / 60.0
        except Exception:
            return pd.to_numeric(val, errors="coerce")
    return pd.to_numeric(val, errors="coerce")


def fetch_scoreboard(date_str: str) -> dict:
    params = {"GameDate": date_str, "LeagueID": "00", "DayOffset": "0"}
    resp = requests.get(SCOREBOARD_URL, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def extract_games(scoreboard: dict) -> List[dict]:
    return scoreboard.get("scoreboard", {}).get("games", []) or []


def fetch_boxscores_for_games(games: Iterable[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    team_frames: List[pd.DataFrame] = []
    player_frames: List[pd.DataFrame] = []

    for game in games:
        game_id = str(game.get("gameId"))
        season_year = int(game.get("seasonYear", dt.datetime.now().year))
        season_id = season_id_from_year(season_year)
        game_date = game.get("gameDate") or game.get("gameDateEst")

        try:
            bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=20)
            dfs = bs.get_data_frames()
            if not dfs:
                print(f"No boxscore frames for game {game_id}")
                continue
            player_df = dfs[0].copy()
            team_df = dfs[1].copy() if len(dfs) > 1 else pd.DataFrame()
        except Exception as exc:
            print(f"Failed to fetch boxscore for {game_id}: {exc}")
            continue

        if not team_df.empty:
            team_df = team_df.rename(
                columns={
                    "TURNOVERS": "TO",
                }
            )
            keep_cols = [
                "GAME_ID",
                "TEAM_ID",
                "TEAM_ABBREVIATION",
                "TEAM_CITY",
                "MIN",
                "FGM",
                "FGA",
                "FG3M",
                "FG3A",
                "FTM",
                "FTA",
                "OREB",
                "DREB",
                "REB",
                "AST",
                "STL",
                "BLK",
                "TO",
                "PF",
                "PTS",
                "PLUS_MINUS",
            ]
            team_df = team_df[[c for c in keep_cols if c in team_df.columns]].copy()
            team_df["SEASON_ID"] = season_id
            team_df["GAME_DATE"] = game_date
            team_df["MIN"] = team_df["MIN"].apply(parse_minutes)
            team_frames.append(team_df)

        if not player_df.empty:
            player_df = player_df.rename(
                columns={
                    "TURNOVERS": "TO",
                    "PERSON_ID": "PLAYER_ID",
                }
            )
            keep_cols = [
                "GAME_ID",
                "TEAM_ID",
                "TEAM_ABBREVIATION",
                "TEAM_CITY",
                "PLAYER_ID",
                "PLAYER_NAME",
                "START_POSITION",
                "COMMENT",
                "MIN",
                "FGM",
                "FGA",
                "FG3M",
                "FG3A",
                "FTM",
                "FTA",
                "OREB",
                "DREB",
                "REB",
                "AST",
                "STL",
                "BLK",
                "TO",
                "PF",
                "PTS",
                "PLUS_MINUS",
            ]
            player_df = player_df[[c for c in keep_cols if c in player_df.columns]].copy()
            player_df["SEASON_ID"] = season_id
            player_df["GAME_DATE"] = game_date
            player_df["MIN"] = player_df["MIN"].apply(parse_minutes)
            player_frames.append(player_df)

    team_out = pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame()
    player_out = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()
    return team_out, player_out


def save_active_players_today(today_games: Iterable[dict]) -> None:
    """Persist today's active rosters (best effort via commonteamroster)."""
    rows: List[Dict] = []
    today = dt.datetime.utcnow().date().isoformat()

    for game in today_games:
        season_year = int(game.get("seasonYear", dt.datetime.now().year))
        season_label = season_label_from_year(season_year)
        for side in ["homeTeam", "awayTeam"]:
            team = game.get(side, {})
            team_id = team.get("teamId")
            team_tri = team.get("teamTricode")
            if not team_id:
                continue
            try:
                roster_df = commonteamroster.CommonTeamRoster(
                    team_id=team_id, season=season_label, timeout=20
                ).get_data_frames()[0]
            except Exception as exc:
                print(f"Roster fetch failed for team {team_id}: {exc}")
                continue

            for _, rec in roster_df.iterrows():
                rows.append(
                    {
                        "GAME_DATE": today,
                        "GAME_ID": game.get("gameId"),
                        "TEAM_ID": team_id,
                        "TEAM_ABBREVIATION": team_tri,
                        "PLAYER_ID": rec.get("PLAYER_ID"),
                        "PLAYER_NAME": rec.get("PLAYER"),
                        "STATUS": rec.get("STATUS"),
                        "IS_ACTIVE": 1,
                    }
                )

    if rows:
        ACTIVE_PLAYERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).drop_duplicates(
            subset=["GAME_DATE", "TEAM_ID", "PLAYER_ID"]
        ).to_csv(ACTIVE_PLAYERS_PATH, index=False)
        print(f"Saved today's active rosters → {ACTIVE_PLAYERS_PATH}")
    else:
        print("No active roster data to save for today.")


def align_with_base_columns(new_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """Reindex new data to base columns, filling missing numeric cols with season means."""
    aligned = new_df.copy()
    base_cols = list(base_df.columns)

    # Compute season means from the most recent season in base data.
    season_year = None
    for col in ["HOME_SEASON_ID", "AWAY_SEASON_ID"]:
        if col in base_df.columns:
            cleaned = base_df[col].dropna()
            if cleaned.empty:
                continue
            try:
                season_year = (
                    cleaned.astype(int)
                    .astype(str)
                    .str[-4:]
                    .astype(int)
                    .max()
                )
            except Exception:
                season_year = None
            if season_year:
                break
    if season_year:
        season_mask = base_df.filter(regex="SEASON_ID").astype(str).apply(
            lambda s: s.str.endswith(str(season_year))
        ).any(axis=1)
        season_means = base_df[season_mask].mean(numeric_only=True)
    else:
        season_means = base_df.mean(numeric_only=True)

    # Add missing columns filled with season averages or 0.
    for col in base_cols:
        if col not in aligned.columns:
            if col in season_means:
                aligned[col] = season_means[col]
            else:
                aligned[col] = 0

    # Drop columns not present in base.
    aligned = aligned[base_cols]

    # Fill missing numeric values with season averages.
    for col in aligned.select_dtypes(include=["number", "bool"]).columns:
        fill_val = season_means.get(col, 0)
        aligned[col] = aligned[col].fillna(fill_val)

    # Simple fill for non-numeric.
    for col in aligned.select_dtypes(exclude=["number", "bool"]).columns:
        aligned[col] = aligned[col].fillna(method="ffill").fillna(method="bfill")
        aligned[col] = aligned[col].fillna("")

    return aligned


def append_to_base(aligned_path: Path, base_path: Path = BASE_PROCESSED) -> int:
    """Append aligned rows into base, return count of new games added."""
    new_df = pd.read_csv(aligned_path, dtype={"GAME_ID": str})
    if new_df.empty:
        return 0

    base_df = pd.read_csv(base_path, dtype={"GAME_ID": str}) if base_path.exists() else pd.DataFrame()
    before = len(base_df)
    combined = pd.concat([base_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["GAME_ID"])
    added = len(combined) - before
    combined.to_csv(base_path, index=False)
    return added


def run_retrain():
    """Re-split and retrain on the updated dataset."""
    print("Re-splitting dataset...")
    split_dataset(
        input_path=BASE_PROCESSED,
        output_dir=PROCESSED_DIR,
        train_frac=0.7,
        val_frac=0.2,
        seed=42,
    )
    print("Retraining models...")
    train_models()


def main():
    today = dt.datetime.utcnow().date()
    yesterday = today - dt.timedelta(days=1)
    yesterday_str = yesterday.isoformat()
    today_str = today.isoformat()

    print(f"Fetching yesterday's games ({yesterday_str}) from decodo scoreboard...")
    try:
        sb_yest = fetch_scoreboard(yesterday_str)
        games_yest = extract_games(sb_yest)
    except Exception as exc:
        print(f"Scoreboard fetch failed for {yesterday_str}: {exc}")
        games_yest = []

    if games_yest:
        print(f"Found {len(games_yest)} games; pulling boxscores...")
        team_df, player_df = fetch_boxscores_for_games(games_yest)
        if not team_df.empty and not player_df.empty:
            TMP_BOX.parent.mkdir(parents=True, exist_ok=True)
            team_df.to_csv(TMP_BOX, index=False)
            player_df.to_csv(TMP_PLAYERS, index=False)
            print(f"Wrote temp boxscores to {TMP_BOX} and {TMP_PLAYERS}")

            # Build processed rows for just the new games.
            run_pipeline(
                box_path=str(TMP_BOX),
                players_path=str(TMP_PLAYERS),
                output_path=str(DAILY_PROCESSED),
            )

            if DAILY_PROCESSED.exists():
                base_df = pd.read_csv(BASE_PROCESSED, dtype={"GAME_ID": str}) if BASE_PROCESSED.exists() else pd.DataFrame()
                new_df = pd.read_csv(DAILY_PROCESSED, dtype={"GAME_ID": str})
                # Keep only games not already in base.
                existing_ids = set(base_df.get("GAME_ID", []))
                new_df = new_df[~new_df["GAME_ID"].astype(str).isin(existing_ids)]
                if not new_df.empty:
                    aligned = align_with_base_columns(new_df, base_df if not base_df.empty else new_df)
                    aligned.to_csv(DAILY_ALIGNED, index=False)
                    added = append_to_base(DAILY_ALIGNED, BASE_PROCESSED)
                    print(f"Appended {added} new games into {BASE_PROCESSED}")
                    if added > 0:
                        run_retrain()
                else:
                    print("No truly new games to append.")
        else:
            print("No boxscore data pulled for yesterday.")
    else:
        print("No games found for yesterday.")

    # Save today's active rosters
    try:
        sb_today = fetch_scoreboard(today_str)
        games_today = extract_games(sb_today)
        if games_today:
            print(f"Saving active rosters for {len(games_today)} games scheduled today...")
            save_active_players_today(games_today)
        else:
            print("No games scheduled today; skipping active roster capture.")
    except Exception as exc:
        print(f"Failed to capture today's active rosters: {exc}")


if __name__ == "__main__":
    main()
