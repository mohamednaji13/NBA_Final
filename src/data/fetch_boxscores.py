import os
import random
import time
from collections import deque, defaultdict
from typing import List, Optional

import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import boxscoretraditionalv3
from dotenv import load_dotenv

from ..utils.logging_config import setup_logging
from ..utils import paths

load_dotenv()
logger = setup_logging(__name__)

BASE_SLEEP_SECONDS = float(os.getenv("NBA_BASE_SLEEP_SECONDS", "1.5"))
MAX_RETRIES_PER_ATTEMPT = int(os.getenv("NBA_MAX_RETRIES", "5"))
BATCH_SAVE_EVERY = int(os.getenv("NBA_BATCH_SAVE_EVERY", "10"))

# Rate limiting controls (to avoid hammering NBA API when using VPN)
RATE_LIMIT_WINDOW_SECONDS = 60
MAX_CALLS_PER_WINDOW = int(os.getenv("NBA_MAX_CALLS_PER_MIN", "8"))
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

def fetch_boxscore_with_retry(game_id: str):
    for attempt in range(1, MAX_RETRIES_PER_ATTEMPT + 1):
        ua = random.choice(USER_AGENTS)
        try:
            _respect_rate_limit()
            kwargs = dict(
                game_id=game_id,
                headers={
                    "User-Agent": ua,
                    "Referer": "https://www.nba.com",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=30,
            )
            if USE_PROXY:
                proxy_url = _next_proxy_url()
                if proxy_url:
                    kwargs["proxy"] = proxy_url

            bs = boxscoretraditionalv3.BoxScoreTraditionalV3(**kwargs)
            df = bs.team_stats.get_data_frame()

            expected_cols = [
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
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = 0

            return df[expected_cols]

        except Exception as e:
            wait = BASE_SLEEP_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                f"[Boxscores V3] GAME_ID={game_id} attempt {attempt}/{MAX_RETRIES_PER_ATTEMPT} "
                f"failed: {e} — retrying in {wait:.1f}s"
            )
            _sleep_with_log(wait, reason=f"retry for GAME_ID={game_id}")

    logger.error(
        f"[Boxscores V3] Exhausted {MAX_RETRIES_PER_ATTEMPT} retries for GAME_ID={game_id}"
    )
    return None

def main():
    if not paths.SCHEDULE_CSV.exists():
        logger.error(f"Schedule CSV not found at {paths.SCHEDULE_CSV}. Run fetch_schedule first.")
        return

    sched = pd.read_csv(paths.SCHEDULE_CSV, dtype={"GAME_ID": str})
    if "GAME_ID" not in sched.columns:
        logger.error("Schedule CSV missing GAME_ID column.")
        return

    unique_ids = sorted(sched["GAME_ID"].dropna().unique())
    queue = deque(unique_ids)
    logger.info(f"Queue initialized with {len(queue)} unique GAME_IDs.")

    all_records = []
    processed_success = 0
    total_attempt_cycles = defaultdict(int)

    pbar = tqdm(total=len(unique_ids), desc="Games (successful)")

    while queue:
        gid = queue.popleft()
        total_attempt_cycles[gid] += 1

        logger.info(
            f"[Boxscores V3] Dequeued GAME_ID={gid} "
            f"(cycle #{total_attempt_cycles[gid]}, remaining in queue={len(queue)})"
        )

        df_team = fetch_boxscore_with_retry(gid)

        if df_team is not None:
            all_records.append(df_team)
            processed_success += 1
            pbar.update(1)

            _sleep_with_log(BASE_SLEEP_SECONDS, reason="between successful games")

            if processed_success % BATCH_SAVE_EVERY == 0 and all_records:
                partial = pd.concat(all_records, ignore_index=True)
                paths.BOXSCORES_PARTIAL_CSV.parent.mkdir(parents=True, exist_ok=True)
                partial.to_csv(paths.BOXSCORES_PARTIAL_CSV, index=False)
                logger.info(
                    f"[Boxscores V3] {processed_success} successful games — "
                    f"partial saved to {paths.BOXSCORES_PARTIAL_CSV}"
                )
        else:
            queue.append(gid)
            logger.warning(
                f"[Boxscores V3] Re-queueing GAME_ID={gid} at end of queue "
                f"(attempt cycles so far: {total_attempt_cycles[gid]})"
            )
            _sleep_with_log(BASE_SLEEP_SECONDS, reason="after requeue")

    pbar.close()

    if not all_records:
        logger.error("No boxscores fetched; exiting.")
        return

    full = pd.concat(all_records, ignore_index=True)
    paths.BOXSCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(paths.BOXSCORES_CSV, index=False)
    logger.info(
        f"Saved full V3 boxscores to {paths.BOXSCORES_CSV} with {len(full)} rows "
        f"from {processed_success} successful games"
    )

if __name__ == "__main__":
    main()
