import sys
from pathlib import Path

import pandas as pd

try:
    from ..utils.logging_config import setup_logging
    from ..utils import paths
except ImportError:
    # Allow running as a script (python3 src/data/merge_raw.py)
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT))
    from src.utils.logging_config import setup_logging  # type: ignore
    from src.utils import paths  # type: ignore

logger = setup_logging(__name__)

PLAYOFF_START_DATES = {
    2019: pd.Timestamp("2019-04-13"),
    2020: pd.Timestamp("2020-08-17"),
    2021: pd.Timestamp("2021-05-22"),
    2022: pd.Timestamp("2022-04-16"),
    2023: pd.Timestamp("2023-04-15"),
    2024: pd.Timestamp("2024-04-20"),
    2025: pd.Timestamp("2025-04-19"),
}


def _read_csv(primary: Path, fallback: Path, dtype: dict) -> pd.DataFrame:
    if primary.exists():
        logger.info(f"Loaded {primary}")
        return pd.read_csv(primary, dtype=dtype)
    if fallback.exists():
        logger.warning(f"{primary.name} not found; using fallback {fallback}")
        return pd.read_csv(fallback, dtype=dtype)
    logger.error(f"Neither {primary} nor {fallback} found.")
    return pd.DataFrame()


def _pair_with_opponent(team_df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per team per game with opponent stats attached."""
    rows = []
    for gid, grp in team_df.sort_values(["GAME_ID", "ORDER_IN_GAME"]).groupby("GAME_ID", sort=False):
        if len(grp) < 2:
            logger.warning(f"GAME_ID={gid} has {len(grp)} teams; skipping opponent pairing.")
            continue
        team_a, team_b = grp.iloc[0], grp.iloc[1]
        for me, opp in [(team_a, team_b), (team_b, team_a)]:
            row = me.to_dict()
            for col, val in opp.items():
                row[f"OPP_{col}"] = val
            rows.append(row)
    return pd.DataFrame(rows)


def _season_from_id(val):
    """Convert SEASON or SEASON_ID into a 4-digit season year."""
    try:
        as_int = int(val)
    except (TypeError, ValueError):
        return None
    # SEASON_ID like 22019 -> 2019
    if as_int > 10000:
        return int(str(as_int)[-4:])
    return as_int


def _add_playoff_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GAME_DATE_DT"] = pd.to_datetime(df.get("GAME_DATE"), errors="coerce")

    season_col = "SEASON" if "SEASON" in df.columns else ("SEASON_ID" if "SEASON_ID" in df.columns else None)
    if season_col:
        df["_SEASON_YEAR"] = df[season_col].apply(_season_from_id)
    else:
        df["_SEASON_YEAR"] = df["GAME_DATE_DT"].dt.year

    def is_playoff(row):
        start = PLAYOFF_START_DATES.get(row["_SEASON_YEAR"])
        if pd.isna(row["GAME_DATE_DT"]) or start is None:
            return pd.NA
        return int(row["GAME_DATE_DT"].date() >= start.date())

    df["IS_PLAYOFF"] = df.apply(is_playoff, axis=1)
    df = df.drop(columns=["_SEASON_YEAR"])
    return df


def _normalize_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric columns (z-score) for training while leaving identifiers untouched.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    for col in ["GAME_ID"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        mean = series.mean()
        std = series.std(ddof=0)
        std = std if pd.notna(std) and std != 0 else 1
        df[col] = (series - mean) / std
    return df


def _fill_missing_numeric(df: pd.DataFrame, fill_value=0) -> pd.DataFrame:
    """Replace NaNs in numeric/bool columns to avoid propagating gaps after merges."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(fill_value)
    return df


def _safe_div(num, denom):
    denom = denom.replace(0, pd.NA) if hasattr(denom, "replace") else denom if denom != 0 else pd.NA
    return num.divide(denom) if hasattr(num, "divide") else (num / denom if denom else pd.NA)


def _add_game_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce").fillna(0)
    df["OPP_PTS"] = pd.to_numeric(df["OPP_PTS"], errors="coerce").fillna(0)
    df["WIN_FLAG"] = (df["PTS"] > df["OPP_PTS"]).astype(int)
    df["MARGIN"] = df["PTS"] - df["OPP_PTS"]
    df["IS_HOME"] = (df.get("ORDER_IN_GAME", 1) == 0).astype(int)

    # Possessions estimate and efficiency
    def poss(form):
        return (
            form["FGA"] - form["OREB"] + form["TO"] + 0.4 * form["FTA"]
        )

    df["POSS_RAW"] = poss(df)
    df["OPP_POSS_RAW"] = poss(df.filter(regex="^OPP_").rename(columns=lambda c: c.replace("OPP_", "")))
    df["POSS_EST"] = 0.5 * (df["POSS_RAW"] + df["OPP_POSS_RAW"])
    df["OFF_RTG"] = 100 * _safe_div(df["PTS"], df["POSS_EST"])
    df["DEF_RTG"] = 100 * _safe_div(df["OPP_PTS"], df["POSS_EST"])
    df["NET_RTG"] = df["OFF_RTG"] - df["DEF_RTG"]
    df["PACE"] = 48 * _safe_div(df["POSS_EST"], df["MIN"] / 5)

    # Shooting profile
    df["EFG"] = _safe_div(df["FGM"] + 0.5 * df["FG3M"], df["FGA"])
    df["TS"] = _safe_div(df["PTS"], 2 * (df["FGA"] + 0.44 * df["FTA"]))
    df["THREE_PAR"] = _safe_div(df["FG3A"], df["FGA"])
    df["FTR"] = _safe_div(df["FTA"], df["FGA"])

    # Rebounding and passing
    df["OREB_PCT"] = _safe_div(df["OREB"], df["OREB"] + df["OPP_DREB"])
    df["DREB_PCT"] = _safe_div(df["DREB"], df["DREB"] + df["OPP_OREB"])
    df["REB_PCT"] = _safe_div(df["REB"], df["REB"] + df["OPP_REB"])
    df["AST_TOV"] = _safe_div(df["AST"], df["TO"])
    df["TOV_PCT"] = _safe_div(df["TO"], df["POSS_EST"])
    df["FT_RATE"] = df["FTR"]

    # Situational flags
    df["CLOSE_WIN_FLAG"] = ((df["MARGIN"].abs() <= 5) & (df["WIN_FLAG"] == 1)).astype(int)
    df["BLOWOUT_WIN_FLAG"] = ((df["MARGIN"].abs() >= 15) & (df["WIN_FLAG"] == 1)).astype(int)
    return df


def _rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GAME_DATE_DT"] = pd.to_datetime(df.get("GAME_DATE"), errors="coerce")
    df = df.sort_values(["TEAM_ID", "GAME_DATE_DT", "GAME_ID"])
    df["GAME_SEQ"] = df.groupby("TEAM_ID").cumcount() + 1

    def roll_mean(name, window):
        return (
            df.groupby("TEAM_ID")[name]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    def roll_sum(name, window):
        return (
            df.groupby("TEAM_ID")[name]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        )

    # Streaks (entering current game)
    def streaks(series):
        vals = []
        streak = 0
        for val in series.shift(1).fillna(0):
            if val == 1:
                streak = streak + 1 if streak >= 0 else 1
            else:
                streak = streak - 1 if streak <= 0 else -1
            vals.append(streak)
        return pd.Series(vals, index=series.index)

    df["WIN_STREAK"] = df.groupby("TEAM_ID")["WIN_FLAG"].transform(streaks)
    df["LOSS_STREAK"] = df["WIN_STREAK"].apply(lambda x: -x if x < 0 else 0)
    df["WIN_STREAK"] = df["WIN_STREAK"].apply(lambda x: x if x > 0 else 0)

    # Rolling win counts/pct
    df["LAST5_WINS"] = roll_sum("WIN_FLAG", 5)
    df["LAST10_WINS"] = roll_sum("WIN_FLAG", 10)
    df["LAST5_WIN_PCT"] = roll_mean("WIN_FLAG", 5)
    df["LAST10_WIN_PCT"] = roll_mean("WIN_FLAG", 10)

    df["SEASON_WIN_PCT"] = (
        df.groupby("TEAM_ID")["WIN_FLAG"]
        .transform(lambda s: s.shift(1).expanding().mean().fillna(0))
    )

    # Margin averages
    df["AVG_MARGIN_SEASON"] = (
        df.groupby("TEAM_ID")["MARGIN"].transform(lambda s: s.shift(1).expanding().mean())
    )
    df["AVG_MARGIN_LAST5"] = roll_mean("MARGIN", 5)
    df["AVG_MARGIN_LAST10"] = roll_mean("MARGIN", 10)

    # Close / blowout win pct
    df["CLOSE_WIN_PCT"] = roll_mean("CLOSE_WIN_FLAG", 10)
    df["BLOWOUT_WIN_PCT"] = roll_mean("BLOWOUT_WIN_FLAG", 10)

    # Opponent history (last 3/5 vs this opponent)
    key = ["TEAM_ID", "OPP_TEAM_ID"]
    df = df.sort_values(["TEAM_ID", "GAME_ID"])
    df["VS_OPP_LAST3"] = (
        df.groupby(key)["WIN_FLAG"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df["VS_OPP_LAST5"] = (
        df.groupby(key)["WIN_FLAG"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # Rolling efficiency
    for col in ["OFF_RTG", "DEF_RTG", "NET_RTG", "PACE", "EFG", "TS", "OREB_PCT", "DREB_PCT", "REB_PCT", "AST_TOV", "TOV_PCT", "FT_RATE", "THREE_PAR", "MARGIN"]:
        df[f"{col}_L5"] = roll_mean(col, 5)
        df[f"{col}_L10"] = roll_mean(col, 10)

    # Rest and schedule-related metrics
    def compute_rest(group: pd.DataFrame) -> pd.DataFrame:
        past_dates = []
        rest_days = []
        b2b = []
        three_in4 = []
        four_in6 = []
        homestand = []
        roadtrip = []
        home_win_pct = []
        away_win_pct = []
        home_games = home_wins = away_games = away_wins = 0
        home_streak = road_streak = 0

        for _, row in group.iterrows():
            dt = row["GAME_DATE_DT"]
            if past_dates and pd.notna(dt) and pd.notna(past_dates[-1]):
                rd = max((dt - past_dates[-1]).days, 0)
            else:
                rd = 0
            rest_days.append(rd)
            b2b.append(1 if past_dates and rd <= 1 else 0)

            if pd.notna(dt):
                count3 = sum(1 for d in past_dates if pd.notna(d) and (dt - d).days <= 3)
                count5 = sum(1 for d in past_dates if pd.notna(d) and (dt - d).days <= 5)
            else:
                count3 = count5 = 0
            three_in4.append(1 if count3 >= 2 else 0)
            four_in6.append(1 if count5 >= 3 else 0)

            is_home = bool(row["IS_HOME"])
            if is_home:
                home_streak += 1
                road_streak = 0
                home_games += 1
                home_wins += row["WIN_FLAG"]
            else:
                road_streak += 1
                home_streak = 0
                away_games += 1
                away_wins += row["WIN_FLAG"]
            homestand.append(home_streak)
            roadtrip.append(road_streak)

            home_win_pct.append(home_wins / home_games if home_games else 0)
            away_win_pct.append(away_wins / away_games if away_games else 0)

            if pd.notna(dt):
                past_dates.append(dt)
            else:
                past_dates.append(pd.NaT)

        group = group.copy()
        group["DAYS_REST"] = rest_days
        group["IS_B2B"] = b2b
        group["IS_3IN4"] = three_in4
        group["IS_4IN6"] = four_in6
        group["HOMESTAND_LEN"] = homestand
        group["ROAD_TRIP_LEN"] = roadtrip
        group["HOME_WIN_PCT"] = home_win_pct
        group["AWAY_WIN_PCT"] = away_win_pct
        return group

    df = df.groupby("TEAM_ID", group_keys=False).apply(compute_rest)

    return df


def _build_wide(team_with_feats: pd.DataFrame, player_block: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = [c for c in team_with_feats.columns if c not in {"GAME_ID", "TEAM_ID"}]

    for gid, grp in team_with_feats.sort_values(["GAME_ID", "ORDER_IN_GAME"]).groupby("GAME_ID", sort=False):
        if len(grp) < 2:
            continue
        team1, team2 = grp.iloc[0], grp.iloc[1]
        row = {"GAME_ID": gid}
        for prefix, rec in [("TEAM1", team1), ("TEAM2", team2)]:
            for col in feature_cols:
                row[f"{prefix}_{col}"] = rec[col]

        # matchup diffs (TEAM1 minus TEAM2)
        row["STREAK_DIFF"] = row.get("TEAM1_WIN_STREAK", 0) - row.get("TEAM2_WIN_STREAK", 0)
        row["LAST10_WIN_PCT_DIFF"] = row.get("TEAM1_LAST10_WIN_PCT", 0) - row.get("TEAM2_LAST10_WIN_PCT", 0)
        row["MARGIN_LAST10_DIFF"] = row.get("TEAM1_AVG_MARGIN_LAST10", 0) - row.get("TEAM2_AVG_MARGIN_LAST10", 0)
        row["NET_RTG_DIFF"] = row.get("TEAM1_NET_RTG", 0) - row.get("TEAM2_NET_RTG", 0)
        row["NET_RTG_L10_DIFF"] = row.get("TEAM1_NET_RTG_L10", 0) - row.get("TEAM2_NET_RTG_L10", 0)
        row["PACE_DIFF"] = row.get("TEAM1_PACE", 0) - row.get("TEAM2_PACE", 0)
        row["EFG_DIFF"] = row.get("TEAM1_EFG", 0) - row.get("TEAM2_EFG", 0)
        row["REB_PCT_DIFF"] = row.get("TEAM1_REB_PCT", 0) - row.get("TEAM2_REB_PCT", 0)
        row["AST_TOV_DIFF"] = row.get("TEAM1_AST_TOV", 0) - row.get("TEAM2_AST_TOV", 0)
        row["THREE_PAR_DIFF"] = row.get("TEAM1_THREE_PAR", 0) - row.get("TEAM2_THREE_PAR", 0)
        row["FTR_DIFF"] = row.get("TEAM1_FTR", 0) - row.get("TEAM2_FTR", 0)
        row["REST_DIFF"] = row.get("TEAM1_DAYS_REST", 0) - row.get("TEAM2_DAYS_REST", 0)

        # Home team margin (positive if home wins, negative if home loses)
        if team1.get("IS_HOME", 0):
            home_margin = team1.get("MARGIN", pd.NA)
        elif team2.get("IS_HOME", 0):
            home_margin = team2.get("MARGIN", pd.NA)
        else:
            # Fallback to points if home marker is missing
            home_margin = team1.get("PTS", 0) - team2.get("PTS", 0)
        row["HOME_MARGIN"] = home_margin

        rows.append(row)

    wide = pd.DataFrame(rows)
    if player_block is not None and not player_block.empty:
        wide = wide.merge(player_block, on="GAME_ID", how="left")
    return wide


def _build_player_block(player_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten player stats so each player becomes its own set of columns per GAME_ID."""
    rows = []
    id_cols = {"GAME_ID", "PLAYER_ID"}
    value_cols = [c for c in player_df.columns if c not in id_cols]

    for gid, grp in player_df.groupby("GAME_ID"):
        row: dict = {"GAME_ID": gid}
        for _, rec in grp.iterrows():
            player_id = rec.get("PLAYER_ID", "")
            prefix = f"PLAYER_{player_id}"
            for col in value_cols:
                row[f"{prefix}_{col}"] = rec[col]
        rows.append(row)

    return pd.DataFrame(rows)


def merge_raw():
    team_df = _read_csv(
        paths.BOXSCORES_CSV, paths.BOXSCORES_PARTIAL_CSV, dtype={"GAME_ID": str}
    )
    player_df = _read_csv(
        paths.BOXSCORES_PLAYERS_CSV, paths.BOXSCORES_PLAYERS_PARTIAL_CSV, dtype={"GAME_ID": str}
    )

    if team_df.empty or player_df.empty:
        logger.error("Team or player boxscores unavailable; aborting merge.")
        return

    team_df = team_df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
    player_df = player_df.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])

    team_df = _add_playoff_flag(team_df)

    # Preserve original order within each game (assume first row is home team).
    team_df["ORDER_IN_GAME"] = team_df.groupby("GAME_ID", sort=False).cumcount()

    paired = _pair_with_opponent(team_df)
    if paired.empty:
        logger.error("No paired team rows; aborting.")
        return

    game_metrics = _add_game_level_metrics(paired)
    with_rolling = _rolling_features(game_metrics)
    player_block = _build_player_block(player_df)

    wide = _build_wide(with_rolling, player_block)
    # Fill numeric gaps (e.g., player stats absent for games the player didn't play) before z-scoring.
    wide_filled = _fill_missing_numeric(wide)
    normalized = _normalize_for_training(wide_filled)

    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = paths.PROCESSED_DIR / "processed.csv"
    out_norm_path = paths.PROCESSED_DIR / "processed_normalized.csv"
    wide.to_csv(out_path, index=False)
    normalized.to_csv(out_norm_path, index=False)
    logger.info(f"Saved merged team+player wide data to {out_path.resolve()}")
    logger.info(f"Saved normalized version for training to {out_norm_path.resolve()}")


def main():
    logger.info("=== Merging team and player boxscores (no schedule) ===")
    merge_raw()


if __name__ == "__main__":
    main()
