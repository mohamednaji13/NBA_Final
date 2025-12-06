import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

import pandas as pd
import requests


def fetch_games_for_date(target_date: str, retries: int = 3, timeout: int = 10):
    url = "https://stats.nba.com/stats/scoreboardv3"
    params = {"GameDate": target_date, "LeagueID": "00", "DayOffset": "0"}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Attempt {attempt}/{retries} failed for {target_date}: {e}")
            if attempt == retries:
                return None
            sleep(2)
    return None


def fetch_games_previous_day(target_date: str | None = None):
    if target_date is None:
        target_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    out_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "previous_day.csv"

    data = fetch_games_for_date(target_date)
    if data is None:
        print(f"Error fetching data for {target_date}.")
        return

    games = data.get("scoreboard", {}).get("games", [])
    if not games:
        print(f"No NBA games on {target_date}.")
        return

    rows = []
    for g in games:
        rows.append(
            {
                "GAME_ID": g["gameId"],
                "SEASON": g.get("seasonYear"),
                "GAME_DATE": target_date,
                "HOME_TEAM_ABBREVIATION": g["homeTeam"]["teamTricode"],
                "AWAY_TEAM_ABBREVIATION": g["awayTeam"]["teamTricode"],
                "HOME_TEAM_ID": g["homeTeam"]["teamId"],
                "AWAY_TEAM_ID": g["awayTeam"]["teamId"],
                "MATCHUP": f"{g['awayTeam']['teamTricode']} @ {g['homeTeam']['teamTricode']}",
                "HOME_SCORE": g["homeTeam"]["score"],
                "AWAY_SCORE": g["awayTeam"]["score"],
                "GAME_STATUS": g["gameStatusText"],
            }
        )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved previous-day schedule to {out_path}")
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch yesterday's NBA games")
    parser.add_argument("--date", help="Override target date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()
    fetch_games_previous_day(args.date)
