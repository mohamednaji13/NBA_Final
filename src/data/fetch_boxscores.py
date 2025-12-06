import os
import random
import time
from collections import deque, defaultdict
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm
from nba_api.stats.endpoints import boxscoretraditionalv3
from dotenv import load_dotenv

from ..utils.logging_config import setup_logging
from ..utils import paths
from nba_api.stats.library import http

load_dotenv()
logger = setup_logging(__name__)

SEASON_PREFIXES = {
    "2019-20": "00219",
    "2020-21": "00220",
    "2021-22": "00221",
    "2022-23": "00222",
    "2023-24": "00223",
    "2024-25": "00224",
    "2025-26": "00225",
}

SEASON_ID_BY_PREFIX = {prefix: int(f"2{season.split('-')[0]}") for season, prefix in SEASON_PREFIXES.items()}

BASE_SLEEP_SECONDS = float(os.getenv("NBA_BASE_SLEEP_SECONDS", "1.5"))
MAX_RETRIES_PER_ATTEMPT = int(os.getenv("NBA_MAX_RETRIES", "5"))
BATCH_SAVE_EVERY = int(os.getenv("NBA_BATCH_SAVE_EVERY", "10"))
MAX_ATTEMPT_CYCLES = int(os.getenv("NBA_MAX_ATTEMPT_CYCLES", "3"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("NBA_REQUEST_TIMEOUT_SECONDS", "45"))
ALLOW_DIRECT_FALLBACK = os.getenv("NBA_ALLOW_DIRECT_FALLBACK", "true").lower() == "true"

# Rate limiting controls (to avoid hammering NBA API when using VPN)
RATE_LIMIT_WINDOW_SECONDS = 60
# Default to a slightly lower call rate to reduce timeouts unless overridden via env.
MAX_CALLS_PER_WINDOW = int(os.getenv("NBA_MAX_CALLS_PER_MIN", "5"))
RATE_LIMIT_JITTER_SECONDS = float(os.getenv("NBA_RATE_JITTER_SECONDS", "0.75"))
CALL_HISTORY = deque()

USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_PORT = os.getenv("PROXY_PORT")
PROXY_USER = os.getenv("PROXY_USER")
PROXY_PASS = os.getenv("PROXY_PASS")
RAW_PROXY_POOL = os.getenv("PROXY_POOL", "")  # comma-separated list of proxy endpoints

PROXY_URL = None
PROXY_POOL_URLS: List[str] = []
PROXY_QUEUE: deque = deque()
PROXY_OPTIONS: List[Optional[str]] = []
SKIP_GAMES: set[str] = set()
SCHEDULE_DATES: dict[str, str] = {}

def _ensure_scheme(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"http://{url}"

def _apply_auth(url: str) -> str:
    if not (PROXY_USER and PROXY_PASS) or "@" in url:
        return url
    if "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://{PROXY_USER}:{PROXY_PASS}@{rest}"
    return url

if USE_PROXY:
    if RAW_PROXY_POOL.strip():
        for part in RAW_PROXY_POOL.split(","):
            cleaned = part.strip()
            if not cleaned:
                continue
            url = _ensure_scheme(cleaned)
            url = _apply_auth(url)
            PROXY_POOL_URLS.append(url)
        if PROXY_POOL_URLS:
            PROXY_QUEUE = deque(PROXY_POOL_URLS)
            logger.info(f"Proxy pool enabled with {len(PROXY_POOL_URLS)} endpoints.")
        else:
            logger.warning("PROXY_POOL provided but no valid entries parsed; falling back.")

    if not PROXY_POOL_URLS and PROXY_HOST and PROXY_PORT and PROXY_USER and PROXY_PASS:
        PROXY_URL = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"
        logger.info(f"Proxy enabled via {PROXY_HOST}:{PROXY_PORT}")

    if not PROXY_POOL_URLS and not PROXY_URL:
        logger.info("Proxy requested but configuration incomplete; calling NBA API directly.")
        USE_PROXY = False
else:
    logger.info("Proxy disabled; calling NBA API directly.")

if USE_PROXY:
    if PROXY_POOL_URLS:
        PROXY_OPTIONS.extend(PROXY_POOL_URLS)
    elif PROXY_URL:
        PROXY_OPTIONS.append(PROXY_URL)
    if ALLOW_DIRECT_FALLBACK:
        PROXY_OPTIONS.append(None)  # allow direct fallback after cycling proxies
elif ALLOW_DIRECT_FALLBACK:
    PROXY_OPTIONS.append(None)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) "
    "Gecko/20100101 Firefox/120.0",
]

V3_TO_OLD_COLS = {
    # identifiers
    "game_id": "GAME_ID",
    "team_id": "TEAM_ID",
    "teamid": "TEAM_ID",
    "team_abbreviation": "TEAM_ABBREVIATION",
    "teamabbreviation": "TEAM_ABBREVIATION",
    "teamtricode": "TEAM_ABBREVIATION",
    "team_city": "TEAM_CITY",
    "teamcity": "TEAM_CITY",
    # scoring / totals
    "points": "PTS",
    "pts": "PTS",
    "rebounds_total": "REB",
    "reboundstotal": "REB",
    "rebounds": "REB",
    "reb": "REB",
    "rebounds_offensive": "OREB",
    "reboundsoffensive": "OREB",
    "oreb": "OREB",
    "rebounds_defensive": "DREB",
    "reboundsdefensive": "DREB",
    "dreb": "DREB",
    "assists": "AST",
    "ast": "AST",
    "steals": "STL",
    "stl": "STL",
    "blocks": "BLK",
    "blk": "BLK",
    "turnovers": "TO",
    "tov": "TO",
    "fouls_personal": "PF",
    "foulspersonal": "PF",
    "personal_fouls": "PF",
    "pf": "PF",
    "field_goals_made": "FGM",
    "fieldgoalsmade": "FGM",
    "fgm": "FGM",
    "field_goals_attempted": "FGA",
    "fieldgoalsattempted": "FGA",
    "fga": "FGA",
    "three_pointers_made": "FG3M",
    "threepointersmade": "FG3M",
    "fg3m": "FG3M",
    "three_pointers_attempted": "FG3A",
    "threepointersattempted": "FG3A",
    "fg3a": "FG3A",
    "free_throws_made": "FTM",
    "freethrowsmade": "FTM",
    "ftm": "FTM",
    "free_throws_attempted": "FTA",
    "freethrowsattempted": "FTA",
    "fta": "FTA",
    "minutes": "MIN",
    "min": "MIN",
    "plus_minus_points": "PLUS_MINUS",
    "plusminus": "PLUS_MINUS",
}

V3_PLAYER_EXTRA_COLS = {
    "player_id": "PLAYER_ID",
    "person_id": "PLAYER_ID",
    "personid": "PLAYER_ID",
    "athlete_id": "PLAYER_ID",
    "player_name": "PLAYER_NAME",
    "athlete_display_name": "PLAYER_NAME",
    "nickname": "PLAYER_NAME",
    "start_position": "START_POSITION",
    "starter": "START_POSITION",
    "position": "START_POSITION",
    "comment": "COMMENT",
}

STAT_COLUMNS = [
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
    "SEASON_ID",
    "GAME_DATE",
]

PLAYER_STAT_COLUMNS = [
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

TEXT_COLS_PLAYER = {"PLAYER_NAME", "START_POSITION", "COMMENT", "TEAM_ABBREVIATION", "TEAM_CITY"}

def _normalize_stats(df: pd.DataFrame, extra_map: dict[str, str], core_stats: List[str]):
    """Normalize V3 columns to legacy names, ensure GAME_ID/TEAM_ID exist, fill missing stats."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    rename_map = {k: v for k, v in V3_TO_OLD_COLS.items() if k in df.columns}
    rename_map.update({k: v for k, v in extra_map.items() if k in df.columns})
    df = df.rename(columns=rename_map)

    if "GAME_ID" not in df.columns and "game_id" in df.columns:
        df["GAME_ID"] = df["game_id"]
    if "TEAM_ID" not in df.columns and "team_id" in df.columns:
        df["TEAM_ID"] = df["team_id"]

    # Ensure core stats exist
    missing_core = [c for c in core_stats if c not in df.columns]
    if missing_core:
        logger.warning(
            f"[Boxscores V3] GAME_ID={df.get('GAME_ID', ['?'])[0] if not df.empty else '?'} missing core stats ({missing_core}); "
            f"columns returned: {list(df.columns)}; filling zeros"
        )
        for c in missing_core:
            df[c] = 0

    if "MIN" in df.columns:
        df["MIN"] = (
            df["MIN"]
            .astype(str)
            .str.extract(r"([0-9]+)")
            .astype(float)
            .fillna(0)
        )

    existing_numeric = [c for c in core_stats if c in df.columns]
    df[existing_numeric] = df[existing_numeric].apply(pd.to_numeric, errors="coerce").fillna(0)

    return df

def _calc_group_plus_minus(pts: pd.Series) -> pd.Series:
    """Compute team plus/minus per GAME_ID using the two team rows."""
    pts = pd.to_numeric(pts, errors="coerce")
    if len(pts) == 2 and pts.notna().all():
        diff = pts.iloc[0] - pts.iloc[1]
        return pd.Series([diff, -diff], index=pts.index)
    return pd.Series([0] * len(pts), index=pts.index)

def _infer_season_id(game_id: str) -> int:
    gid_str = str(game_id).zfill(10)
    return SEASON_ID_BY_PREFIX.get(gid_str[:5], 0)

def _attach_team_derivatives(df_team: pd.DataFrame, game_id: str) -> pd.DataFrame:
    df_team = df_team.copy()
    df_team["PTS"] = pd.to_numeric(df_team["PTS"], errors="coerce").fillna(0)
    pm = (
        df_team.groupby("GAME_ID", sort=False)["PTS"]
        .apply(_calc_group_plus_minus)
        .reset_index(level=0, drop=True)
    )
    df_team["PLUS_MINUS"] = pm.fillna(0).astype(int)
    df_team["SEASON_ID"] = _infer_season_id(game_id)
    df_team["GAME_DATE"] = SCHEDULE_DATES.get(str(game_id), "")
    return df_team

def _sleep_with_log(seconds: float, reason: str = ""):
    if seconds <= 0:
        return
    if reason:
        logger.info(f"Sleeping {seconds:.1f}s ({reason})")
    time.sleep(seconds)

def _respect_rate_limit():
    """Ensure we do not exceed MAX_CALLS_PER_WINDOW within the last minute."""
    if MAX_CALLS_PER_WINDOW <= 0:
        return

    now = time.monotonic()
    # Drop timestamps outside the window
    while CALL_HISTORY and now - CALL_HISTORY[0] > RATE_LIMIT_WINDOW_SECONDS:
        CALL_HISTORY.popleft()

    if len(CALL_HISTORY) >= MAX_CALLS_PER_WINDOW:
        wait = RATE_LIMIT_WINDOW_SECONDS - (now - CALL_HISTORY[0])
        wait += random.uniform(0, RATE_LIMIT_JITTER_SECONDS)
        _sleep_with_log(wait, reason="rate limit pacing")
        now = time.monotonic()
        while CALL_HISTORY and now - CALL_HISTORY[0] > RATE_LIMIT_WINDOW_SECONDS:
            CALL_HISTORY.popleft()

    CALL_HISTORY.append(time.monotonic())

def _next_proxy_url() -> Optional[str]:
    if PROXY_QUEUE:
        url = PROXY_QUEUE[0]
        PROXY_QUEUE.rotate(-1)
        return url
    return PROXY_URL

def _select_proxy_for_attempt(attempt: int) -> Optional[str]:
    """Choose proxy (or None for direct) for this attempt."""
    if ALLOW_DIRECT_FALLBACK:
        if attempt == 1:
            # Try direct first to avoid proxy-related blocking/stripping.
            return None
        if attempt == MAX_RETRIES_PER_ATTEMPT:
            # Ensure the final retry always attempts a direct call in case proxies are blocked.
            return None
    if not PROXY_OPTIONS:
        return _next_proxy_url()
    idx = (attempt - 1) % len(PROXY_OPTIONS)
    return PROXY_OPTIONS[idx]

def _set_session_proxy(proxy_url: Optional[str]):
    """Override nba_api HTTP session to enforce proxy usage when provided."""
    session = requests.Session()
    if proxy_url:
        session.proxies.update({"http": proxy_url, "https": proxy_url})
    http._session = session  # type: ignore[attr-defined]

def fetch_boxscore_with_retry(game_id: str):
    use_proxy_after_failure = USE_PROXY  # option C: only switch to proxy when needed
    for attempt in range(1, MAX_RETRIES_PER_ATTEMPT + 1):
        ua = random.choice(USER_AGENTS)
        try:
            _respect_rate_limit()
            proxy_url = None
            if use_proxy_after_failure:
                proxy_url = _select_proxy_for_attempt(attempt)
                if proxy_url is None and USE_PROXY and ALLOW_DIRECT_FALLBACK:
                    logger.info(f"[Boxscores V3] Attempt {attempt}: falling back to direct (no proxy)")
            _set_session_proxy(proxy_url)
            headers = {
                "Host": "stats.nba.com",
                "Connection": "keep-alive",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": ua,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Origin": "https://www.nba.com",
                "Referer": "https://www.nba.com/",
                "Sec-Fetch-Site": "same-site",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Dest": "empty",
                "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not:A-Brand";v="99"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "X-NBA-Stats-Origin": "stats",
                "X-NBA-Stats-Token": "true",
            }

            kwargs = dict(
                game_id=game_id,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if proxy_url:
                kwargs["proxy"] = proxy_url

            bs = boxscoretraditionalv3.BoxScoreTraditionalV3(**kwargs)
            df_team = bs.team_stats.get_data_frame()
            df_player = bs.player_stats.get_data_frame()

            df_team = _normalize_stats(df_team, {}, [
                "PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TO", "PF",
                "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "MIN"
            ])
            df_player = _normalize_stats(df_player, V3_PLAYER_EXTRA_COLS, [
                "PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TO", "PF",
                "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "MIN"
            ])

            if df_team.empty or df_player.empty:
                raise ValueError("team_stats or player_stats returned empty DataFrame")

            if "GAME_ID" not in df_team.columns:
                logger.warning("team_stats missing GAME_ID column; injecting from requested GAME_ID")
                df_team["GAME_ID"] = game_id
            if "GAME_ID" not in df_player.columns:
                logger.warning("player_stats missing GAME_ID column; injecting from requested GAME_ID")
                df_player["GAME_ID"] = game_id

            # Ensure we did not get a bad/blocked response that nba_api still parsed.
            game_id_str = str(game_id)
            df_game_ids = df_team["GAME_ID"].astype(str).unique()
            if game_id_str not in df_game_ids:
                raise ValueError(f"Unexpected GAME_IDs returned in team_stats: {df_game_ids}")

            if df_team["GAME_ID"].nunique() == 0 or df_team[[
                "PTS", "REB", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA"
            ]].sum().sum() == 0:
                raise ValueError("team_stats returned all zero values (likely bad response)")

            df_team = _attach_team_derivatives(df_team, game_id_str)

            for c in STAT_COLUMNS:
                if c not in df_team.columns:
                    df_team[c] = 0
            for c in PLAYER_STAT_COLUMNS:
                if c not in df_player.columns:
                    df_player[c] = "" if c in TEXT_COLS_PLAYER else 0

            return df_team[STAT_COLUMNS], df_player[PLAYER_STAT_COLUMNS]

        except Exception as e:
            wait = BASE_SLEEP_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                f"[Boxscores V3] GAME_ID={game_id} attempt {attempt}/{MAX_RETRIES_PER_ATTEMPT} "
                f"failed: {e} — retrying in {wait:.1f}s"
            )
            # If we failed once and proxy is available, enable proxy for subsequent attempts (option C)
            if USE_PROXY:
                use_proxy_after_failure = True
            _sleep_with_log(wait, reason=f"retry for GAME_ID={game_id}")

    logger.error(
        f"[Boxscores V3] Exhausted {MAX_RETRIES_PER_ATTEMPT} retries for GAME_ID={game_id}"
    )
    return None, None

def main():
    if not paths.SCHEDULE_CSV.exists():
        logger.error(f"Schedule CSV not found at {paths.SCHEDULE_CSV}. Run fetch_schedule first.")
        return

    sched = pd.read_csv(paths.SCHEDULE_CSV, dtype={"GAME_ID": str})
    if "GAME_ID" not in sched.columns:
        logger.error("Schedule CSV missing GAME_ID column.")
        return

    if "SEASON" not in sched.columns:
        logger.error("Schedule CSV missing SEASON column required for prefix validation.")
        return

    sched["SEASON_ID"] = sched["SEASON"].astype(str)

    valid_ids = []
    bad_ids = []

    for _, row in sched.iterrows():
        prefix = SEASON_PREFIXES.get(row["SEASON_ID"])
        gid = str(row["GAME_ID"])

        if prefix and gid.startswith(prefix) and len(gid) == 10:
            valid_ids.append(gid)
        else:
            bad_ids.append(gid)

    if bad_ids:
        logger.info(f"Dropped {len(bad_ids)} invalid GAME_ID(s) failing prefix validation.")

    # Map GAME_ID -> GAME_DATE for stamping team boxscores
    if "GAME_DATE" in sched.columns:
        SCHEDULE_DATES.update(sched.set_index("GAME_ID")["GAME_DATE"].to_dict())

    queue = deque(sorted(set(valid_ids)))
    unique_ids = list(queue)
    logger.info(f"Queue initialized with {len(queue)} unique GAME_IDs.")

    all_team_records = []
    all_player_records = []
    processed_success = 0
    total_attempt_cycles = defaultdict(int)
    skipped_ids = []

    pbar = tqdm(total=len(unique_ids), desc="Games (successful)", position=0, dynamic_ncols=True, leave=True)
    remaining_bar = tqdm(
        total=len(queue),
        desc="Games remaining",
        position=1,
        dynamic_ncols=True,
        leave=True,
    )
    previous_remaining = len(queue)

    while queue:
        gid = queue.popleft()
        total_attempt_cycles[gid] += 1

        logger.info(
            f"[Boxscores V3] Dequeued GAME_ID={gid} "
            f"(cycle #{total_attempt_cycles[gid]}, remaining in queue={len(queue)})"
        )

        df_team, df_players = fetch_boxscore_with_retry(gid)

        if df_team is not None and df_players is not None:
            all_team_records.append(df_team)
            all_player_records.append(df_players)
            processed_success += 1
            pbar.update(1)

            _sleep_with_log(BASE_SLEEP_SECONDS, reason="between successful games")

            if processed_success % BATCH_SAVE_EVERY == 0 and all_team_records:
                partial_team = pd.concat(all_team_records, ignore_index=True)
                partial_players = pd.concat(all_player_records, ignore_index=True)
                paths.BOXSCORES_PARTIAL_CSV.parent.mkdir(parents=True, exist_ok=True)
                partial_team.to_csv(paths.BOXSCORES_PARTIAL_CSV, index=False)
                paths.BOXSCORES_PLAYERS_PARTIAL_CSV.parent.mkdir(parents=True, exist_ok=True)
                partial_players.to_csv(paths.BOXSCORES_PLAYERS_PARTIAL_CSV, index=False)
                logger.info(
                    f"[Boxscores V3] {processed_success} successful games — "
                    f"partial saved to {paths.BOXSCORES_PARTIAL_CSV} and players to {paths.BOXSCORES_PLAYERS_PARTIAL_CSV}"
                )
        else:
            if total_attempt_cycles[gid] >= MAX_ATTEMPT_CYCLES:
                skipped_ids.append(gid)
                logger.error(
                    f"[Boxscores V3] Skipping GAME_ID={gid} after {total_attempt_cycles[gid]} cycles."
                )
            else:
                queue.append(gid)
                logger.warning(
                    f"[Boxscores V3] Re-queueing GAME_ID={gid} at end of queue "
                    f"(attempt cycles so far: {total_attempt_cycles[gid]})"
                )
                _sleep_with_log(BASE_SLEEP_SECONDS, reason="after requeue")

        current_remaining = len(queue)
        delta = previous_remaining - current_remaining
        if delta > 0:
            remaining_bar.update(delta)
        remaining_bar.set_postfix_str(f"{current_remaining} left")
        previous_remaining = current_remaining

    pbar.close()
    remaining_bar.close()

    if not all_team_records and not skipped_ids:
        logger.error("No boxscores fetched; exiting.")
        return

    if all_team_records:
        full_team = pd.concat(all_team_records, ignore_index=True)
        full_players = pd.concat(all_player_records, ignore_index=True)
        paths.BOXSCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
        full_team.to_csv(paths.BOXSCORES_CSV, index=False)
        full_players.to_csv(paths.BOXSCORES_PLAYERS_CSV, index=False)
        logger.info(
            f"Saved full V3 team boxscores to {paths.BOXSCORES_CSV} with {len(full_team)} rows "
            f"from {processed_success} successful games"
        )
        logger.info(
            f"Saved full V3 player boxscores to {paths.BOXSCORES_PLAYERS_CSV} with {len(full_players)} rows "
            f"from {processed_success} successful games"
        )

    if skipped_ids:
        logger.error(f"Skipped {len(skipped_ids)} GAME_IDs after exhausting cycles: {', '.join(skipped_ids[:10])}" + ("..." if len(skipped_ids) > 10 else ""))

if __name__ == "__main__":
    main()
