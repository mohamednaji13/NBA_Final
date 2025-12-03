from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

from .. import config
from ..utils.logging_config import setup_logging
from ..utils import paths

logger = setup_logging(__name__)

def fetch_season(season: str) -> pd.DataFrame:
    logger.info(f"Fetching schedule for season {season}")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
    )
    df = gamefinder.get_data_frames()[0]
    if "SEASON_TYPE" in df.columns:
        df = df[df["SEASON_TYPE"] == "Regular Season"]
    df["SEASON_LABEL"] = season
    return df

def main():
    all_dfs = []
    for season in config.LAST_FIVE_SEASONS:
        try:
            df = fetch_season(season)
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch {season}: {e}")

    if not all_dfs:
        logger.error("No seasons fetched; exiting.")
        return

    full = pd.concat(all_dfs, ignore_index=True)
    paths.SCHEDULE_CSV.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(paths.SCHEDULE_CSV, index=False)
    logger.info(f"Saved schedule to {paths.SCHEDULE_CSV} with {len(full)} rows")

if __name__ == "__main__":
    main()
