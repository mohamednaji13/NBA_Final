import pandas as pd

from ..utils.logging_config import setup_logging
from ..utils import paths

logger = setup_logging(__name__)

ROLLING_WINDOWS = (3, 5, 10)
ROLLING_COLS = [
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TO",
    "PF",
    "PLUS_MINUS",
    "OREB",
    "DREB",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
]
SEASON_AVG_COLS = ["PTS", "REB", "AST", "EFG", "TS"]

def detect_side_from_matchup(matchup: str) -> str:
    if isinstance(matchup, str):
        if "vs." in matchup:
            return "HOME"
        if "@" in matchup:
            return "AWAY"
    return "UNKNOWN"

def add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    denom_efg = df["FGA"].replace(0, pd.NA)
    denom_ts = (df["FGA"] + 0.44 * df["FTA"]).replace(0, pd.NA)
    df["EFG"] = ((df["FGM"] + 0.5 * df["FG3M"]) / denom_efg).fillna(0)
    df["TS"] = (df["PTS"] / (2 * denom_ts)).fillna(0)
    return df

def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    df["REST_DAYS"] = (
        df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(99)
    )
    df["IS_BACK_TO_BACK"] = (df["REST_DAYS"] <= 1).astype(int)
    return df

def compute_streaks(results: pd.Series) -> pd.Series:
    streaks = []
    current = 0
    prev = None
    for res in results:
        streaks.append(current)
        if res == "W":
            current = current + 1 if prev == "W" else 1
            prev = "W"
        elif res == "L":
            current = current - 1 if prev == "L" else -1
            prev = "L"
        else:
            current = 0
            prev = None
    return pd.Series(streaks, index=results.index)

def add_streak_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "WL" not in df.columns:
        df["STREAK"] = 0
        logger.warning("WL column missing; streak set to 0.")
        return df
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    df["STREAK"] = (
        df.groupby("TEAM_ID")["WL"]
        .apply(compute_streaks)
        .reset_index(level=0, drop=True)
    )
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    for col in ROLLING_COLS:
        if col not in df.columns:
            logger.warning(f"Column {col} missing; skipping rolling calcs.")
            continue
        for window in ROLLING_WINDOWS:
            df[f"{col}_roll{window}"] = (
                df.groupby("TEAM_ID")[col]
                .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
    return df

def add_season_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    for col in SEASON_AVG_COLS:
        if col not in df.columns:
            logger.warning(f"Column {col} missing; skipping season averages.")
            continue
        df[f"{col}_season_avg"] = (
            df.groupby(["TEAM_ID", "SEASON"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
    return df

def main():
    if not paths.SCHEDULE_CSV.exists() or not paths.BOXSCORES_CSV.exists():
        logger.error("Missing raw CSVs; run fetch_schedule and fetch_boxscores first.")
        return

    sched = pd.read_csv(paths.SCHEDULE_CSV, dtype={"GAME_ID": str})
    box = pd.read_csv(paths.BOXSCORES_CSV, dtype={"GAME_ID": str})

    if "GAME_ID" not in sched.columns or "TEAM_ID" not in sched.columns:
        logger.error("Schedule CSV must have GAME_ID and TEAM_ID.")
        return

    logger.info("Merging schedule and V3 boxscores on GAME_ID + TEAM_ID...")
    team_games = sched.merge(
        box,
        on=["GAME_ID", "TEAM_ID"],
        how="inner",
        suffixes("", "_BOX"),
    )

    if "GAME_DATE" in team_games.columns:
        team_games["GAME_DATE"] = pd.to_datetime(team_games["GAME_DATE"])
    else:
        logger.error("GAME_DATE not found in schedule.")
        return

    if "SEASON" in team_games.columns:
        team_games["SEASON"] = team_games["SEASON"].astype(str)
    elif "SEASON_LABEL" in team_games.columns:
        team_games["SEASON"] = team_games["SEASON_LABEL"].astype(str)
    else:
        logger.error("No SEASON column found; expected SEASON or SEASON_LABEL.")
        return

    if "MATCHUP" not in team_games.columns:
        logger.error("MATCHUP column missing; cannot determine HOME/AWAY.")
        return

    team_games["SIDE"] = team_games["MATCHUP"].apply(detect_side_from_matchup)
    team_games = team_games[team_games["SIDE"].isin(["HOME", "AWAY"])]

    team_games = add_efficiency_features(team_games)
    team_games = add_rest_features(team_games)
    team_games = add_streak_feature(team_games)
    team_games = add_rolling_features(team_games)
    team_games = add_season_averages(team_games)

    logger.info("Pivoting to game-level with HOME_*/AWAY_* features...")
    home = team_games[team_games["SIDE"] == "HOME"].copy()
    away = team_games[team_games["SIDE"] == "AWAY"].copy()

    home = home.add_prefix("HOME_")
    away = away.add_prefix("AWAY_")

    dataset = home.merge(
        away,
        left_on="HOME_GAME_ID",
        right_on="AWAY_GAME_ID",
        how="inner",
        suffixes("", "_AWAY"),
    )

    dataset["GAME_ID"] = dataset["HOME_GAME_ID"]
    dataset["GAME_DATE"] = dataset["HOME_GAME_DATE"]
    dataset["SEASON"] = dataset["HOME_SEASON"]
    dataset["HOME_HOME_COURT"] = 1
    dataset["AWAY_HOME_COURT"] = 0

    if "HOME_WL" in dataset.columns:
        dataset["home_team_won"] = (dataset["HOME_WL"] == "W").astype(int)
    else:
        logger.error("HOME_WL column not found after pivot; cannot create target.")
        return

    if "HOME_PTS" in dataset.columns and "AWAY_PTS" in dataset.columns:
        dataset["score_margin"] = dataset["HOME_PTS"] - dataset["AWAY_PTS"]

    if "HOME_REST_DAYS" in dataset.columns and "AWAY_REST_DAYS" in dataset.columns:
        dataset["rest_diff"] = dataset["HOME_REST_DAYS"] - dataset["AWAY_REST_DAYS"]

    if "HOME_STREAK" in dataset.columns and "AWAY_STREAK" in dataset.columns:
        dataset["streak_diff"] = dataset["HOME_STREAK"] - dataset["AWAY_STREAK"]

    if "HOME_EFG" in dataset.columns and "AWAY_EFG" in dataset.columns:
        dataset["efg_diff"] = dataset["HOME_EFG"] - dataset["AWAY_EFG"]
    if "HOME_TS" in dataset.columns and "AWAY_TS" in dataset.columns:
        dataset["ts_diff"] = dataset["HOME_TS"] - dataset["AWAY_TS"]

    # Opponent stat differentials using recent form (5-game rolling if present)
    if "HOME_PTS_roll5" in dataset.columns and "AWAY_PTS_roll5" in dataset.columns:
        dataset["pts_roll5_diff"] = dataset["HOME_PTS_roll5"] - dataset["AWAY_PTS_roll5"]
    if "HOME_REB_roll5" in dataset.columns and "AWAY_REB_roll5" in dataset.columns:
        dataset["reb_roll5_diff"] = dataset["HOME_REB_roll5"] - dataset["AWAY_REB_roll5"]
    if "HOME_AST_roll5" in dataset.columns and "AWAY_AST_roll5" in dataset.columns:
        dataset["ast_roll5_diff"] = dataset["HOME_AST_roll5"] - dataset["AWAY_AST_roll5"]

    # Save
    paths.DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(paths.DATASET_CSV, index=False)
    logger.info(
        f"Saved dataset to {paths.DATASET_CSV} with {len(dataset)} rows and {dataset.shape[1]} columns"
    )

if __name__ == "__main__":
    main()
