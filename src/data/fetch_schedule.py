import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from ..utils.logging_config import setup_logging
from ..utils import paths

logger = setup_logging(__name__)

# Seasons you chose:
SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

def fetch_season_games(season):
    """
    Uses official leaguegamelog endpoint to get REAL GAME_IDs.
    Only regular season + playoffs.
    """
    logger.info(f"Fetching season {season}...")

    log = leaguegamelog.LeagueGameLog(
        league_id="00",    # NBA
        season=season,
        season_type_all_star="Regular Season"
    ).get_data_frames()[0]

    playoffs = leaguegamelog.LeagueGameLog(
        league_id="00",
        season=season,
        season_type_all_star="Playoffs"
    ).get_data_frames()[0]

    df = pd.concat([log, playoffs], ignore_index=True)

    # Normalize columns
    df = df.rename(columns={
        "GAME_ID": "GAME_ID",
        "GAME_DATE": "GAME_DATE",
        "MATCHUP": "MATCHUP",
        "TEAM_ID": "TEAM_ID",
        "WL": "WL"
    })

    # Add season label for downstream use
    df["SEASON"] = season

    # Convert GAME_DATE into datetime
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    return df


def resolve_team_ids(df):
    """
    Stable, fast TEAM_ID resolver that never blocks and never loops infinitely.
    Uses TEAM_ABBREVIATION → TEAM_ID mapping from df itself.
    """
    # Build mapping from available rows
    mapping = (
        df[['TEAM_ABBREVIATION', 'TEAM_ID']]
        .dropna()
        .drop_duplicates()
        .set_index('TEAM_ABBREVIATION')['TEAM_ID']
        .to_dict()
    )

    # Apply mapping safely
    df['HOME_TEAM_ID'] = df['HOME_TEAM_ABBREVIATION'].map(mapping)
    df['AWAY_TEAM_ID'] = df['AWAY_TEAM_ABBREVIATION'].map(mapping)

    # Fill missing IDs with -1 (unknown)
    df['HOME_TEAM_ID'] = df['HOME_TEAM_ID'].fillna(-1).astype(int)
    df['AWAY_TEAM_ID'] = df['AWAY_TEAM_ID'].fillna(-1).astype(int)

    return df


def build_schedule():
    all_games = []

    for season in SEASONS:
        try:
            season_df = fetch_season_games(season)
            all_games.append(season_df)
        except Exception as e:
            logger.error(f"Failed season {season}: {e}")

    df = pd.concat(all_games, ignore_index=True)

    # Split MATCHUP → home / away
    # MATCHUP looks like: "LAL vs PHX" or "BOS @ MIA"
    def parse(matchup):
        if " vs " in matchup:
            home, away = matchup.split(" vs ")
            return home, away
        elif " @ " in matchup:
            away, home = matchup.split(" @ ")
            return home, away
        return None, None

    df["HOME_TEAM_ABBREVIATION"], df["AWAY_TEAM_ABBREVIATION"] = zip(*df["MATCHUP"].map(parse))

    logger.info("Resolving TEAM_IDs for home/away...")
    df = resolve_team_ids(df)

    # Sort final schedule
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Save
    paths.SCHEDULE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.SCHEDULE_CSV, index=False)

    logger.info(
        f"Schedule complete → {len(df)} games saved to {paths.SCHEDULE_CSV}"
    )


if __name__ == "__main__":
    build_schedule()
