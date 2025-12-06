import os
import sys
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from nba_api.stats.endpoints import commonteamroster
from pydantic import BaseModel

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

TARGET_COL = "POINT_DIFF"
WINDOW_LABELS = [1, 2, 3, 4, None]

SCHEDULE_CSV = ROOT / "execution" / "schedule.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "data.csv"
MODEL_DIR = ROOT / "models_store"
PREDICTIONS_DIR = ROOT / "predictions"

app = FastAPI(title="NBA Point Diff Predictor")
static_dir = ROOT / "execution" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class PredictRequest(BaseModel):
    home_team_id: int
    away_team_id: int


def load_latest_model():
    models = sorted(MODEL_DIR.glob("*_best_model.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not models:
        raise HTTPException(status_code=500, detail="No model found in models_store")
    return joblib.load(models[0])


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


def load_team_options() -> pd.DataFrame:
    raw_schedule = ROOT / "data" / "raw" / "games_schedule.csv"
    source = None
    if raw_schedule.exists():
        try:
            source = pd.read_csv(
                raw_schedule,
                dtype={"HOME_TEAM_ID": str, "HOME_TEAM_ABBREVIATION": str, "HOME_TEAM_CITY": str},
            )
        except Exception:
            source = None
    if source is None:
        source = pd.read_csv(
            PROCESSED_CSV,
            dtype={"HOME_TEAM_ID": str, "HOME_TEAM_ABBREVIATION": str, "HOME_TEAM_CITY": str},
        )

    # Ensure required columns exist
    for col in ["HOME_TEAM_ID", "HOME_TEAM_ABBREVIATION", "HOME_TEAM_CITY"]:
        if col not in source.columns:
            source[col] = ""

    teams = source[["HOME_TEAM_ID", "HOME_TEAM_ABBREVIATION", "HOME_TEAM_CITY"]].drop_duplicates()
    teams = teams.rename(
        columns={
            "HOME_TEAM_ID": "TEAM_ID",
            "HOME_TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
            "HOME_TEAM_CITY": "TEAM_CITY",
        }
    )
    teams["DISPLAY_NAME"] = teams["TEAM_CITY"].astype(str) + " " + teams["TEAM_ABBREVIATION"].astype(str)
    return teams.sort_values("DISPLAY_NAME")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    teams = load_team_options()
    options = "".join(f"<option value='{row.TEAM_ID}'>{row.DISPLAY_NAME}</option>" for _, row in teams.iterrows())
    html = f"""
    <html>
    <head>
      <title>NBA Point Diff Predictor</title>
      <style>
        body {{ font-family: Arial, sans-serif; background: #0b1021; color: #e1e7ef; margin: 0; padding: 0; }}
        .container {{ max-width: 640px; margin: 60px auto; background: #12182d; border-radius: 12px; padding: 32px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }}
        h2 {{ margin-top: 0; }}
        label {{ display: block; margin: 12px 0 6px; font-weight: 600; }}
        select, button {{ width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #28314a; background: #0f1427; color: #e1e7ef; }}
        button {{ margin-top: 16px; background: linear-gradient(135deg, #3a7bd5, #00d2ff); border: none; cursor: pointer; font-weight: 700; }}
        button:hover {{ filter: brightness(1.05); }}
        a {{ color: #00d2ff; }}
      </style>
    </head>
    <body>
      <div class="container">
        <h2>NBA Point Diff Predictor</h2>
        <p>Select a matchup to get the model's predicted point differential and winner.</p>
        <form action="/predict" method="post">
          <label>Home Team</label>
          <select name="home_team_id">{options}</select>
          <label>Away Team</label>
          <select name="away_team_id">{options}</select>
          <button type="submit">Predict</button>
        </form>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.post("/predict", response_class=HTMLResponse)
async def predict_game(request: Request):
    form = await request.form()
    home_team_id = form.get("home_team_id")
    away_team_id = form.get("away_team_id")
    if not home_team_id or not away_team_id:
        return HTMLResponse("Home and away teams are required", status_code=400)

    home_team_id = int(home_team_id)
    away_team_id = int(away_team_id)

    # Load model and processed data
    model = load_latest_model()
    model_features = getattr(model, "feature_names_in_", None)
    processed = pd.read_csv(PROCESSED_CSV)
    processed["GAME_DATE"] = pd.to_datetime(processed.get("GAME_DATE", pd.Timestamp.utcnow()))
    processed["_SEASON_YEAR"] = add_season_year(processed)
    max_season = processed["_SEASON_YEAR"].dropna().max()

    if model_features is not None:
        feature_cols = list(model_features)
    else:
        feature_cols = (
            processed.drop(columns=[TARGET_COL], errors="ignore")
            .select_dtypes(include=["number", "bool"])
            .filter(regex="^(?!.*PLUS_MINUS).*$")
            .columns.tolist()
        )

    # Pick best window
    window_scores = []
    for win in WINDOW_LABELS:
        if pd.isna(max_season):
            allowed = None
        else:
            allowed = None if win is None else list(range(int(max_season - win + 1), int(max_season) + 1))
        mse = evaluate_window(processed, feature_cols, model, allowed)
        window_scores.append((win, mse))
    best_win, best_mse = min(window_scores, key=lambda t: t[1])

    allowed = None if pd.isna(max_season) else (None if best_win is None else list(range(int(max_season - best_win + 1), int(max_season) + 1)))

    filtered = processed if allowed is None else processed[processed["_SEASON_YEAR"].isin(allowed)]
    global_means = filtered.mean(numeric_only=True)
    home_means = compute_team_means(filtered, "HOME_", allowed)
    away_means = compute_team_means(filtered, "AWAY_", allowed)
    home_player_means = compute_player_means(filtered, "HOME_", allowed)
    away_player_means = compute_player_means(filtered, "AWAY_", allowed)

    roster_cache: Dict[int, Set[str]] = {}
    season_label = str(int(max_season)) if not pd.isna(max_season) else None
    skip_roster = os.environ.get("SKIP_ROSTER", "").lower() in {"1", "true", "yes"}
    for tid in [home_team_id, away_team_id]:
        if skip_roster:
            roster_cache[tid] = set()
            continue
        try:
            if season_label:
                roster_df = commonteamroster.CommonTeamRoster(
                    team_id=int(tid), season=season_label, timeout=5
                ).get_data_frames()[0]
                roster_cache[tid] = set(roster_df["PLAYER_ID"].astype(str))
            else:
                roster_cache[tid] = set()
        except Exception:
            roster_cache[tid] = set()

    feat_row = build_feature_row(
        feature_cols,
        home_team_id,
        away_team_id,
        home_means,
        away_means,
        global_means,
        roster_cache.get(home_team_id, set()),
        roster_cache.get(away_team_id, set()),
        home_player_means,
        away_player_means,
    )

    features_df = pd.DataFrame([feat_row]).reindex(columns=feature_cols, fill_value=0).fillna(0)
    X = features_df.astype(float)
    pred = float(model.predict(X)[0])
    winner = "HOME" if pred > 0 else "AWAY"

    html = f"""
    <html>
    <head>
      <style>
        body {{ font-family: Arial, sans-serif; background: #0b1021; color: #e1e7ef; }}
        .container {{ max-width: 640px; margin: 60px auto; background: #12182d; border-radius: 12px; padding: 32px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }}
        a {{ color: #00d2ff; }}
      </style>
    </head>
    <body>
      <div class="container">
        <h3>Prediction</h3>
        <p><strong>Home ID:</strong> {home_team_id} | <strong>Away ID:</strong> {away_team_id}</p>
        <p><strong>Point Diff:</strong> {pred:.2f}</p>
        <p><strong>Winner:</strong> {winner}</p>
        <a href="/">Back</a>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html)
