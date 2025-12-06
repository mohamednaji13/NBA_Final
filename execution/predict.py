import os
import sys
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set

import joblib
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import commonteamroster

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from execution.fetch_games_daily import fetch_games_daily  # type: ignore

TARGET_COL = "POINT_DIFF"
WINDOW_LABELS = [1, 2, 3, 4, None]  # None = all seasons

SCHEDULE_CSV = ROOT / "execution" / "schedule.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "data.csv"
MODEL_DIR = ROOT / "models_store"
PREDICTIONS_DIR = ROOT / "predictions"
OUTPUT_MD = PREDICTIONS_DIR / "predictions_today.md"


def load_latest_model():
    models = sorted(MODEL_DIR.glob("*_best_model.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not models:
        raise FileNotFoundError(f"No model files found in {MODEL_DIR}")
    print(f"Using model: {models[0].name}")
    return joblib.load(models[0])


def todays_games(schedule: pd.DataFrame) -> pd.DataFrame:
    schedule = schedule.copy()
    schedule["GAME_DATE"] = pd.to_datetime(schedule["GAME_DATE"], utc=True, errors="coerce").dt.tz_localize(None)
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    return schedule[schedule["GAME_DATE"] == today].sort_values("GAME_DATE")


def load_schedule_today() -> pd.DataFrame:
    # Refresh schedule file for today
    fetch_games_daily()
    if not SCHEDULE_CSV.exists():
        return pd.DataFrame()
    sched = pd.read_csv(SCHEDULE_CSV, dtype={"GAME_ID": str})
    window = todays_games(sched)
    if not window.empty:
        print(f"Using schedule: {SCHEDULE_CSV.name}")
    return window


def season_year_from_id(val) -> Optional[int]:
    try:
        s = str(int(val))
        return int(s[-4:])
    except Exception:
        return None


def add_season_year(df: pd.DataFrame) -> pd.Series:
    for col in [c for c in df.columns if c.endswith("SEASON_ID")]:
        series = df[col].apply(season_year_from_id)
        if series.notna().any():
            return series
    return pd.Series([None] * len(df))


def compute_team_means(df: pd.DataFrame, prefix: str, allowed_seasons: Optional[List[int]]) -> Dict[int, pd.Series]:
    team_col = f"{prefix}TEAM_ID"
    cols = [c for c in df.columns if c.startswith(prefix)]
    filtered = df if not allowed_seasons else df[df["_SEASON_YEAR"].isin(allowed_seasons)]
    means = {}
    for team_id, grp in filtered.groupby(team_col):
        means[int(team_id)] = grp[cols].mean(numeric_only=True)
    return means


def compute_player_means(df: pd.DataFrame, prefix: str, allowed_seasons: Optional[List[int]]) -> Dict[str, pd.Series]:
    filtered = df if not allowed_seasons else df[df["_SEASON_YEAR"].isin(allowed_seasons)]
    player_cols = [c for c in filtered.columns if c.startswith(f"{prefix}PLAYER_")]
    by_player: DefaultDict[str, List[str]] = DefaultDict(list)
    for col in player_cols:
        parts = col.split("_")
        if len(parts) < 3:
            continue
        player_id = parts[2]
        by_player[player_id].append(col)

    means: Dict[str, pd.Series] = {}
    for pid, cols in by_player.items():
        means[pid] = filtered[cols].mean(numeric_only=True)
    return means


def evaluate_window(processed: pd.DataFrame, feature_cols: List[str], model, allowed_seasons: Optional[List[int]], holdout_n: int = 500) -> float:
    filtered = processed if allowed_seasons is None else processed[processed["_SEASON_YEAR"].isin(allowed_seasons)]
    if filtered.empty or TARGET_COL not in filtered:
        return np.inf
    subset = filtered.tail(holdout_n)
    col_means = filtered[feature_cols].mean(numeric_only=True)
    X = subset[feature_cols].reindex(columns=feature_cols, fill_value=0)
    X = X.fillna(col_means).fillna(0).astype(float)
    y_true = subset[TARGET_COL]
    y_pred = model.predict(X)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return mse


def build_feature_row(
    feature_cols: List[str],
    home_team_id: int,
    away_team_id: int,
    home_means: Dict[int, pd.Series],
    away_means: Dict[int, pd.Series],
    global_means: pd.Series,
    home_roster: Set[str],
    away_roster: Set[str],
    home_player_means: Dict[str, pd.Series],
    away_player_means: Dict[str, pd.Series],
) -> Dict:
    row = {}
    home_series = home_means.get(home_team_id)
    away_series = away_means.get(away_team_id)

    for col in feature_cols:
        if col.startswith("HOME_"):
            if col.startswith("HOME_PLAYER_"):
                player_id = col.split("_")[2]
                p_mean = home_player_means.get(player_id)
                if player_id not in home_roster and p_mean is not None:
                    val = p_mean.get(col, global_means.get(col, 0))
                elif player_id not in home_roster:
                    val = global_means.get(col, 0)
                elif p_mean is not None:
                    val = p_mean.get(col, global_means.get(col, 0))
                else:
                    val = None if home_series is None else home_series.get(col)
            else:
                val = None if home_series is None else home_series.get(col)
            if pd.isna(val):
                val = global_means.get(col, 0)
            row[col] = val if not pd.isna(val) else 0
        elif col.startswith("AWAY_"):
            if col.startswith("AWAY_PLAYER_"):
                player_id = col.split("_")[2]
                p_mean = away_player_means.get(player_id)
                if player_id not in away_roster and p_mean is not None:
                    val = p_mean.get(col, global_means.get(col, 0))
                elif player_id not in away_roster:
                    val = global_means.get(col, 0)
                elif p_mean is not None:
                    val = p_mean.get(col, global_means.get(col, 0))
                else:
                    val = None if away_series is None else away_series.get(col)
            else:
                val = None if away_series is None else away_series.get(col)
            if pd.isna(val):
                val = global_means.get(col, 0)
            row[col] = val if not pd.isna(val) else 0
        else:
            val = global_means.get(col, 0)
            row[col] = val if not pd.isna(val) else 0
    return row


def main() -> None:
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"Processed feature file not found: {PROCESSED_CSV}")

    window = load_schedule_today()
    if window.empty:
        print("No games found for today.")
        return

    model = load_latest_model()
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        feature_cols = list(model_features)
    else:
        processed_tmp = pd.read_csv(PROCESSED_CSV)
        processed_tmp["GAME_DATE"] = pd.to_datetime(processed_tmp.get("GAME_DATE", window["GAME_DATE"].min()))
        feature_cols = (
            processed_tmp.drop(columns=[TARGET_COL], errors="ignore")
            .select_dtypes(include=["number", "bool"])
            .filter(regex="^(?!.*PLUS_MINUS).*$")
            .columns.tolist()
        )

    processed = pd.read_csv(PROCESSED_CSV)
    processed["GAME_DATE"] = pd.to_datetime(processed.get("GAME_DATE", window["GAME_DATE"].min()))
    processed["_SEASON_YEAR"] = add_season_year(processed)
    max_season = processed["_SEASON_YEAR"].dropna().max()

    window_scores = []
    for win in WINDOW_LABELS:
        if pd.isna(max_season):
            allowed = None
        else:
            allowed = None if win is None else list(range(int(max_season - win + 1), int(max_season) + 1))
        mse = evaluate_window(processed, feature_cols, model, allowed)
        window_scores.append((win, mse))
    best_win, best_mse = min(window_scores, key=lambda t: t[1])
    print("Window scores (win, mse):", window_scores)
    print(f"Best window: {best_win or 'all'} (MSE={best_mse:.4f})")

    if pd.isna(max_season):
        allowed = None
    else:
        allowed = None if best_win is None else list(range(int(max_season - best_win + 1), int(max_season) + 1))

    filtered = processed if allowed is None else processed[processed["_SEASON_YEAR"].isin(allowed)]
    global_means = filtered.mean(numeric_only=True)

    home_means = compute_team_means(filtered, "HOME_", allowed)
    away_means = compute_team_means(filtered, "AWAY_", allowed)
    home_player_means = compute_player_means(filtered, "HOME_", allowed)
    away_player_means = compute_player_means(filtered, "AWAY_", allowed)

    roster_cache: Dict[int, Set[str]] = {}
    season_label = str(int(max_season)) if not pd.isna(max_season) else None
    skip_roster = os.environ.get("SKIP_ROSTER", "").lower() in {"1", "true", "yes"}
    for team_id in pd.concat([window["HOME_TEAM_ID"], window["AWAY_TEAM_ID"]]).dropna().unique():
        if skip_roster:
            roster_cache[int(team_id)] = set()
            continue
        try:
            if season_label:
                roster_df = commonteamroster.CommonTeamRoster(
                    team_id=int(team_id), season=season_label, timeout=5
                ).get_data_frames()[0]
                roster_cache[int(team_id)] = set(roster_df["PLAYER_ID"].astype(str))
            else:
                roster_cache[int(team_id)] = set()
        except Exception:
            roster_cache[int(team_id)] = set()

    rows = []
    for _, game in window.iterrows():
        home_id = int(game.get("HOME_TEAM_ID", -1))
        away_id = int(game.get("AWAY_TEAM_ID", -1))
        feat_row = build_feature_row(
            feature_cols,
            home_id,
            away_id,
            home_means,
            away_means,
            global_means,
            roster_cache.get(home_id, set()),
            roster_cache.get(away_id, set()),
            home_player_means,
            away_player_means,
        )
        rows.append(feat_row)

    features_df = pd.DataFrame(rows).reindex(columns=feature_cols, fill_value=0).fillna(0)
    X = features_df.astype(float)
    preds = model.predict(X)

    out = window.reset_index(drop=True).copy()
    out["PRED_POINT_DIFF"] = preds
    out["PRED_WINNER"] = out["PRED_POINT_DIFF"].apply(lambda p: "HOME" if p > 0 else "AWAY")

    md = [
        f"# Best window: {best_win or 'all'} (MSE={best_mse:.4f})",
        out[["GAME_ID", "GAME_DATE", "MATCHUP", "PRED_POINT_DIFF", "PRED_WINNER"]].to_markdown(index=False),
    ]
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n\n".join(md))
    print(f"Saved predictions using best window to {OUTPUT_MD.resolve()}")


if __name__ == "__main__":
    main()
