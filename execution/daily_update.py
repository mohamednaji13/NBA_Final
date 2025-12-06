"""
daily_update.py — DEBUG MODE ENABLED
----------------------------------------------
Run this version ONCE to reveal the real error.
After error is fixed, DEBUG_MODE will be turned OFF.
----------------------------------------------
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import requests
import datetime as dt
import time
from pathlib import Path
import sys
import warnings

# Enable debugging
DEBUG_MODE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from merge import run_pipeline  # noqa

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DATA_COPY = PROCESSED_DIR / "data_copy.csv"
FINAL_OUT = RAW_DIR / "final_daily_update.csv"

TMP_BOX = RAW_DIR / "boxscores_yesterday.csv"
TMP_PLAYERS = RAW_DIR / "boxscores_players_yesterday.csv"
TMP_MERGED = PROCESSED_DIR / "data_daily_temp.csv"

SCHEDULE_FALLBACK = ROOT / "execution" / "schedule.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

# Stats.nba.com blocks bare requests; add the common headers used by their site.
NBA_STATS_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "x-nba-stats-token": "true",
    "x-nba-stats-origin": "stats",
}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _get_with_retries(url, *, params=None, headers=None, timeout=20, retries=3, backoff=5):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                if DEBUG_MODE:
                    print(f"DEBUG: retry {attempt}/{retries} for {url} ({exc})")
                time.sleep(backoff)
    if last_exc:
        raise last_exc
    return None


def parse_min(x):
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    if ":" in s:
        try:
            m, ss = s.split(":")
            return float(m) + float(ss) / 60.0
        except:
            return 0.0
    return 0.0


def load_master_dataset():
    if not DATA_COPY.exists():
        raise FileNotFoundError(f"Missing data_copy.csv at {DATA_COPY}")
    df = pd.read_csv(DATA_COPY)
    return df, list(df.columns)


# ---------------------------------------------------------
# SCOREBOARD FETCH
# ---------------------------------------------------------
def _schedule_games(date_str):
    """Fallback: use local schedule.csv to build game list for a date."""
    if not SCHEDULE_FALLBACK.exists():
        return []
    try:
        df = pd.read_csv(SCHEDULE_FALLBACK)
        df["GAME_DATE"] = df["GAME_DATE"].astype(str)
        day_games = df[df["GAME_DATE"] == date_str]
    except Exception:
        return []

    games = []
    for _, r in day_games.iterrows():
        games.append({
            "gameId": str(r["GAME_ID"]),
            "gameDate": date_str,
            "homeTeam": {
                "teamId": int(r["HOME_TEAM_ID"]),
                "teamTricode": r.get("HOME_TEAM_ABBREVIATION", "") or "",
                "teamCity": "",
            },
            "awayTeam": {
                "teamId": int(r["AWAY_TEAM_ID"]),
                "teamTricode": r.get("AWAY_TEAM_ABBREVIATION", "") or "",
                "teamCity": "",
            },
        })
    return games


def _merge_games(primary, secondary):
    """Append games from secondary when their gameId is missing in primary."""
    seen = {g.get("gameId") for g in primary}
    extra = [g for g in secondary if g.get("gameId") not in seen]
    return primary + extra


def fetch_scoreboard(date_str):
    params = {"GameDate": date_str, "LeagueID": "00", "DayOffset": "0"}
    is_today = date_str == dt.date.today().isoformat()

    # Try official
    try:
        r = _get_with_retries(
            "https://stats.nba.com/stats/scoreboardv3",
            params=params,
            headers=NBA_STATS_HEADERS,
            timeout=60,
            retries=4,
            backoff=5,
        )
        js = r.json() if r is not None else {}
        games = extract_games(js)
        if games:
            if DEBUG_MODE: print(f"DEBUG: Official scoreboard success ({len(games)} games)")
            return js
    except Exception as e:
        if DEBUG_MODE: print("DEBUG: Official scoreboard failed:", e)

    # Local schedule fallback (preferred when official fails)
    sched_games = _schedule_games(date_str)
    if sched_games:
        if DEBUG_MODE:
            print(f"DEBUG: Using local schedule fallback ({len(sched_games)} games)")
        return {"scoreboard": {"games": sched_games}}

    # CDN only exposes "today" – use it only for current day if schedule unavailable
    if is_today:
        try:
            url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
            r = _get_with_retries(
                url,
                headers=HEADERS,
                timeout=30,
                retries=4,
                backoff=5,
            )
            js = r.json() if r is not None else {}
            games = extract_games(js)
            if DEBUG_MODE:
                print(f"DEBUG: CDN scoreboard success ({len(games)} games)")
            return {"scoreboard": {"games": games}}
        except Exception as e:
            if DEBUG_MODE: print("DEBUG: CDN scoreboard failed:", e)

    return {"scoreboard": {"games": []}}


def extract_games(sb):
    return sb.get("scoreboard", {}).get("games", []) or []


# ---------------------------------------------------------
# EWMA BUILDER
# ---------------------------------------------------------
def build_team_ewma(history_df, alpha=0.20):
    numeric = history_df.select_dtypes(include="number").columns
    ewma_map = {}

    teams = (
        set(history_df["HOME_TEAM_ID"].dropna().astype(int)) |
        set(history_df["AWAY_TEAM_ID"].dropna().astype(int))
    )

    for tid in teams:
        sub = history_df[(history_df["HOME_TEAM_ID"] == tid) |
                         (history_df["AWAY_TEAM_ID"] == tid)]
        if sub.empty:
            continue
        ewma_map[tid] = sub[numeric].ewm(alpha=alpha).mean().iloc[-1].to_dict()

    league_avg = history_df[numeric].mean().to_dict()
    return ewma_map, league_avg, list(numeric)


def build_league_team_defaults(history_df):
    """
    Build simple league-average defaults for core box score stats,
    derived from both home and away team columns when available.
    """
    stats = [
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "REB", "AST", "STL", "BLK",
        "TO", "PF", "PTS", "PLUS_MINUS", "MIN"
    ]
    defaults = {s: 0.0 for s in stats}
    for stat in stats:
        cols = []
        home_col = f"HOME_{stat}_TEAM"
        away_col = f"AWAY_{stat}_TEAM"
        if home_col in history_df.columns:
            cols.append(home_col)
        if away_col in history_df.columns:
            cols.append(away_col)
        if cols:
            defaults[stat] = history_df[cols].mean().mean()

    # Reasonable minute fallback if absent
    defaults["MIN"] = defaults.get("MIN", 240.0) or 240.0
    return defaults


# ---------------------------------------------------------
# FETCH YESTERDAY MERGED
# ---------------------------------------------------------
def fetch_yesterday_merged(master_df):
    today = dt.date.today()
    yday = (today - dt.timedelta(days=1)).isoformat()

    if DEBUG_MODE: print(f"DEBUG: Fetching yesterday scoreboard: {yday}")

    sb = fetch_scoreboard(yday)
    games = extract_games(sb)

    if DEBUG_MODE: print(f"DEBUG: Found {len(games)} yesterday games")

    if not games:
        return pd.DataFrame([])

    from nba_api.stats.endpoints import boxscoretraditionalv3, commonteamroster

    ewma, league_avg, numeric_cols = build_team_ewma(master_df)
    league_defaults = build_league_team_defaults(master_df)

    team_out = []
    player_out = []

    def fallback_team(team_id, gid, gdate):
        row = {
            "GAME_ID": gid,
            "TEAM_ID": team_id,
            "GAME_DATE": gdate,
            "SEASON_ID": 0,
            "TEAM_ABBREVIATION": "",
            "TEAM_CITY": "",
            "TEAM_NAME": "",
        }
        for stat, val in league_defaults.items():
            row[stat] = val
        return pd.DataFrame([row])

    def fallback_players(team_id, gid, gdate):
        roster = pd.DataFrame([{"PLAYER_ID": -1, "PLAYER": "SYNTH_PLAYER"}])
        for attempt in range(1, 4):
            try:
                roster = commonteamroster.CommonTeamRoster(
                    team_id=team_id,
                    season="2024-25"
                ).get_data_frames()[0]
                break
            except Exception as e:
                if DEBUG_MODE: print(f"DEBUG: Roster attempt {attempt} failed:", e)
                time.sleep(2)

        rows = []
        for _, r in roster.iterrows():
            rows.append({
                "GAME_ID": gid,
                "TEAM_ID": team_id,
                "PLAYER_ID": r.get("PLAYER_ID", -1),
                "PLAYER_NAME": r.get("PLAYER", "SYNTH"),
                "START_POSITION": r.get("START_POSITION", ""),
                "COMMENT": "",
                "MIN": league_defaults.get("MIN", 240.0) / max(len(roster), 1),
                "FGM": league_defaults.get("FGM", 0),
                "FGA": league_defaults.get("FGA", 0),
                "FG3M": league_defaults.get("FG3M", 0),
                "FG3A": league_defaults.get("FG3A", 0),
                "FTM": league_defaults.get("FTM", 0),
                "FTA": league_defaults.get("FTA", 0),
                "OREB": league_defaults.get("OREB", 0),
                "DREB": league_defaults.get("DREB", 0),
                "REB": league_defaults.get("REB", 0),
                "AST": league_defaults.get("AST", 0),
                "STL": league_defaults.get("STL", 0),
                "BLK": league_defaults.get("BLK", 0),
                "TO": league_defaults.get("TO", 0),
                "PF": league_defaults.get("PF", 0),
                "PTS": league_defaults.get("PTS", 0),
                "PLUS_MINUS": league_defaults.get("PLUS_MINUS", 0),
                "GAME_DATE": gdate,
                "SEASON_ID": 0
            })
        return pd.DataFrame(rows)

    for g in games:
        gid = g["gameId"]
        home = g["homeTeam"]["teamId"]
        away = g["awayTeam"]["teamId"]
        gdate = g.get("gameDate") or g.get("gameDateEst")

        if DEBUG_MODE:
            print(f"DEBUG: Fetching boxscore for {gid}")

        dfs = []
        for attempt in range(1, 4):
            try:
                bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid)
                dfs = bs.get_data_frames()
                break
            except Exception as e:
                if DEBUG_MODE: print(f"DEBUG: Boxscore attempt {attempt} failed:", e)
                time.sleep(3)
        if not dfs:
            if DEBUG_MODE: print("DEBUG: Boxscore failed after retries")

        # If boxscore missing
        if not dfs or dfs[0].empty:
            if DEBUG_MODE: print("DEBUG: Using fallback team & players")
            player_out.append(fallback_players(home, gid, gdate))
            player_out.append(fallback_players(away, gid, gdate))
            team_out.append(fallback_team(home, gid, gdate))
            team_out.append(fallback_team(away, gid, gdate))
            continue

        # PLAYER FRAME
        p = dfs[0].rename(columns={"TURNOVERS": "TO", "PERSON_ID": "PLAYER_ID"})
        p["GAME_ID"] = gid
        if "MIN" in p.columns:
            p["MIN"] = p["MIN"].apply(parse_min)
        else:
            p["MIN"] = 0
        for c in numeric_cols:
            if c not in p.columns:
                p[c] = 0
        player_out.append(p)

        # TEAM FRAME
        if len(dfs) > 1 and not dfs[1].empty:
            t = dfs[1].rename(columns={"TURNOVERS": "TO"})
            t["GAME_ID"] = gid

            if len(t) >= 2:
                t.loc[t.index[0], "TEAM_ID"] = away
                t.loc[t.index[1], "TEAM_ID"] = home
            elif len(t) == 1:
                t.loc[t.index[0], "TEAM_ID"] = home

            if "MIN" in t.columns:
                t["MIN"] = t["MIN"].apply(parse_min)

            for c in numeric_cols:
                if c not in t.columns:
                    t[c] = 0

            team_out.append(t)
        else:
            team_out.append(fallback_team(home, gid, gdate))
            team_out.append(fallback_team(away, gid, gdate))

    # -----------------------
    # RUN MERGE PIPELINE
    # -----------------------
    try:
        if player_out:
            pd.concat(player_out, ignore_index=True).to_csv(TMP_PLAYERS, index=False)
        else:
            # Ensure a minimal players file exists to keep pipeline running
            pd.DataFrame([{
                "GAME_ID": None, "TEAM_ID": None, "PLAYER_ID": -1, "PLAYER_NAME": "SYNTH",
                "MIN": 0, "FGM": 0, "FGA": 0, "FG3M": 0, "FG3A": 0,
                "FTM": 0, "FTA": 0, "OREB": 0, "DREB": 0, "REB": 0,
                "AST": 0, "STL": 0, "BLK": 0, "TO": 0, "PF": 0,
                "PTS": 0, "PLUS_MINUS": 0,
            }]).to_csv(TMP_PLAYERS, index=False)

        if team_out:
            pd.concat(team_out, ignore_index=True).to_csv(TMP_BOX, index=False)

        run_pipeline(str(TMP_BOX), str(TMP_PLAYERS), str(TMP_MERGED))
        return pd.read_csv(TMP_MERGED)
    except Exception as e:
        if DEBUG_MODE:
            print("DEBUG: merge pipeline failed:", e)
        return pd.DataFrame([])


# ---------------------------------------------------------
# TODAY'S EWMA PREDICTION ROWS
# ---------------------------------------------------------
def fetch_today_rows(master_df, schema):
    today = dt.date.today().isoformat()

    if DEBUG_MODE:
        print(f"DEBUG: Fetching today scoreboard: {today}")

    sb = fetch_scoreboard(today)
    games = extract_games(sb)

    if DEBUG_MODE:
        print(f"DEBUG: Today games found: {len(games)}")

    if not games:
        return pd.DataFrame(columns=schema)

    ewma, league_avg, numeric_cols = build_team_ewma(master_df)

    rows = []

    for g in games:
        gid = g["gameId"]
        home_id = int(g["homeTeam"]["teamId"])
        away_id = int(g["awayTeam"]["teamId"])
        gdate = g.get("gameDate") or g.get("gameDateEst")

        home_info = g.get("homeTeam", {})
        away_info = g.get("awayTeam", {})

        home_vec = ewma.get(home_id, league_avg)
        away_vec = ewma.get(away_id, league_avg)

        row = {col: 0 for col in schema}

        row["GAME_ID"] = gid
        row["GAME_DATE"] = gdate
        row["HOME_GAME_ID"] = gid
        row["AWAY_GAME_ID"] = gid
        row["HOME_TEAM_ID"] = home_id
        row["AWAY_TEAM_ID"] = away_id
        row["HOME_TEAM_ABBREVIATION"] = home_info.get("teamTricode", "")
        row["AWAY_TEAM_ABBREVIATION"] = away_info.get("teamTricode", "")
        row["HOME_TEAM_CITY"] = home_info.get("teamCity", "")
        row["AWAY_TEAM_CITY"] = away_info.get("teamCity", "")

        # Fill stat features from EWMA maps (skip ID/meta fields)
        skip_meta = {
            "HOME_TEAM_ID", "AWAY_TEAM_ID",
            "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION",
            "HOME_TEAM_CITY", "AWAY_TEAM_CITY",
            "HOME_GAME_ID", "AWAY_GAME_ID",
            "HOME_GAME_DATE", "AWAY_GAME_DATE",
            "GAME_ID", "GAME_DATE",
        }

        for col in schema:
            if col in skip_meta:
                continue
            if col.startswith("HOME_"):
                base = col.replace("HOME_", "")
                row[col] = home_vec.get(base, 0)
            elif col.startswith("AWAY_"):
                base = col.replace("AWAY_", "")
                row[col] = away_vec.get(base, 0)
            elif col.startswith("DIFF_"):
                base = col.replace("DIFF_", "")
                row[col] = home_vec.get(base, 0) - away_vec.get(base, 0)

        row["POINT_DIFF"] = row.get("DIFF_PTS", 0)
        row["TARGET_HOME_WIN"] = 1 if row["POINT_DIFF"] > 0 else 0

        rows.append(row)

    return pd.DataFrame(rows)[schema]


# ---------------------------------------------------------
# MAIN (DEBUG MODE ENABLED)
# ---------------------------------------------------------
def main():
    try:
        if DEBUG_MODE: print("DEBUG: Loading master dataset…")

        master, schema = load_master_dataset()

        if DEBUG_MODE: print("DEBUG: Fetching yesterday merged rows…")
        y = fetch_yesterday_merged(master)

        if DEBUG_MODE: print("DEBUG: Fetching today prediction rows…")
        t = fetch_today_rows(master, schema)

        added = len(y) + len(t)

        if DEBUG_MODE:
            print(f"DEBUG: Yesterday rows: {len(y)}, Today rows: {len(t)}")

        if added > 0:
            combined = pd.concat([master, y, t], ignore_index=True)
            # Ensure every schema column exists; fill missing with 0
            combined = combined.reindex(columns=schema, fill_value=0)
            combined.to_csv(DATA_COPY, index=False)
            combined.to_csv(FINAL_OUT, index=False)

        if DEBUG_MODE:
            print("DEBUG: SUCCESS. Added rows:", added)
        else:
            print(f"Daily update complete ({added} rows added)")

    except Exception as e:
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
            print("\nDEBUG: Fatal Error:", e)
        else:
            print("Daily update failed")


if __name__ == "__main__":
    main()
