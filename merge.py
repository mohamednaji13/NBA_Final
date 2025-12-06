import pandas as pd
import numpy as np

# ============================================================
#          COMMENT → MEANINGFUL INTEGER ENCODING
# ============================================================

def encode_comment(value):
    """Map COMMENT text to a meaningful integer category."""
    if pd.isna(value) or value.strip() == "":
        return 0  # Played normally or no note

    v = value.lower()

    # Various DNP categories
    if "dnp" in v:
        if "coach" in v:
            return -1  # Coach's Decision
        if "injury" in v or "illness" in v:
            return -2  # Medical DNP
        return -3      # Other DNP (rest, inactive)

    # General inactive cases
    if "inactive" in v or "did not dress" in v or "dnd" in v:
        return -3

    # Suspension / ejection
    if "suspend" in v or "eject" in v:
        return -4

    return 1  # Played with a non-DNP note


# ============================================================
#                LOAD RAW TEAM & PLAYER DATA
# ============================================================

def load_data(box_path, players_path):
    box = pd.read_csv(box_path)
    players = pd.read_csv(players_path)

    box.columns = box.columns.str.upper()
    players.columns = players.columns.str.upper()

    return box, players


# ============================================================
#        AGGREGATE PLAYER STATS → TEAM STATS (1 row/team/game)
# ============================================================

def aggregate_player_to_team(players):
    agg_funcs = {
        "PTS": "sum",
        "REB": "sum",
        "AST": "sum",
        "STL": "sum",
        "BLK": "sum",
        "TO": "sum",
        "FGA": "sum",
        "FGM": "sum",
        "FG3A": "sum",
        "FG3M": "sum",
        "FTA": "sum",
        "FTM": "sum",
        "OREB": "sum",
        "DREB": "sum",
        "PLUS_MINUS": "sum",
        "MIN": "sum"
    }

    team = players.groupby(["GAME_ID", "TEAM_ID"]).agg(agg_funcs).reset_index()

    # Advanced team shooting
    team["TS_PCT"] = team["PTS"] / (2 * (team["FGA"] + 0.44 * team["FTA"]).replace(0, 1))
    team["EFG_PCT"] = (team["FGM"] + 0.5 * team["FG3M"]) / team["FGA"].replace(0, 1)
    team["FTR"] = team["FTA"] / team["FGA"].replace(0, 1)
    team["THREEPAR"] = team["FG3A"] / team["FGA"].replace(0, 1)

    return team


# ============================================================
#          MERGE TEAM BOXSCORE + AGGREGATED TEAM STATS
# ============================================================

def merge_team_context(box, team_stats):
    df = box.merge(team_stats, on=["GAME_ID", "TEAM_ID"], suffixes=("_TEAM", "_PLYR"), how="left")
    return df


# ============================================================
#          PLAYER-LEVEL WIDE BLOCK (1 row/game, cols per player)
# ============================================================

def build_player_block(players):
    """
    Flatten player boxscore rows so each player becomes a set of columns
    prefixed with PLAYER_<PLAYER_ID>_*. One row per GAME_ID.
    """
    rows = []
    id_cols = {"GAME_ID", "PLAYER_ID"}
    value_cols = [c for c in players.columns if c not in id_cols]

    for gid, grp in players.groupby("GAME_ID"):
        row = {"GAME_ID": gid}
        for _, rec in grp.iterrows():
            pid = rec.get("PLAYER_ID", "")
            prefix = f"PLAYER_{pid}"
            for col in value_cols:
                row[f"{prefix}_{col}"] = rec[col]
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
#              HOME/AWAY ASSIGNMENT + GAME MERGE
# ============================================================

def pivot_to_games(df):
    """
    Converts 2 team-rows per game into 1 row per game.
    Rule: first team = HOME, second team = AWAY.
    """

    # Ensure consistent ordering
    df = df.sort_values(["GAME_ID", "TEAM_ID"]).copy()

    # First row = home, second = away
    df["TEAM_ORDER"] = df.groupby("GAME_ID").cumcount()

    home = df[df["TEAM_ORDER"] == 0].copy().add_prefix("HOME_")
    away = df[df["TEAM_ORDER"] == 1].copy().add_prefix("AWAY_")

    games = home.merge(
        away,
        left_on="HOME_GAME_ID",
        right_on="AWAY_GAME_ID",
        how="inner"
    ).drop(columns=["AWAY_GAME_ID"])

    return games


# ============================================================
#                GAME-LEVEL ENGINEERED FEATURES
# ============================================================

def resolve_stat_col(df, prefix, base):
    """Pick the right column name given TEAM/PLYR suffixes."""
    for suffix in ("_PLYR", "_TEAM", ""):
        candidate = f"{prefix}_{base}{suffix}"
        if candidate in df.columns:
            return candidate
    raise KeyError(f"No column found for {prefix}_{base} (checked _PLYR, _TEAM, none)")


def compute_game_level_features(g):
    g = g.copy()

    diff_cols = [
        "PTS", "REB", "AST", "STL", "BLK",
        "TS_PCT", "EFG_PCT",
        "FGA", "FG3A", "FTA",
        "OREB", "DREB", "TO"
    ]

    for col in diff_cols:
        home_col = resolve_stat_col(g, "HOME", col)
        away_col = resolve_stat_col(g, "AWAY", col)
        g[f"DIFF_{col}"] = g[home_col] - g[away_col]

    # Target label: home win (use the same source as diffs)
    pts_home = resolve_stat_col(g, "HOME", "PTS")
    pts_away = resolve_stat_col(g, "AWAY", "PTS")
    g["TARGET_HOME_WIN"] = (g[pts_home] > g[pts_away]).astype(int)
    g["POINT_DIFF"] = g[pts_home] - g[pts_away]  # positive if home scores more

    return g


# ============================================================
#            ADVANCED NBA METRICS (PACE, RTG, SHOOTING)
# ============================================================

def safe_div(num, denom):
    return num / denom.replace(0, np.nan)


def add_advanced_metrics(g):
    g = g.copy()

    def val(prefix, base):
        return pd.to_numeric(g[resolve_stat_col(g, prefix, base)], errors="coerce")

    for home_pref, away_pref in [("HOME", "AWAY"), ("AWAY", "HOME")]:
        fga = val(home_pref, "FGA")
        fgm = val(home_pref, "FGM")
        fg3a = val(home_pref, "FG3A")
        fg3m = val(home_pref, "FG3M")
        fta = val(home_pref, "FTA")
        oreb = val(home_pref, "OREB")
        dreb = val(home_pref, "DREB")
        reb = val(home_pref, "REB")
        ast = val(home_pref, "AST")
        tov = val(home_pref, "TO")
        pts = val(home_pref, "PTS")
        opp_pts = val(away_pref, "PTS")
        opp_oreb = val(away_pref, "OREB")
        opp_dreb = val(away_pref, "DREB")

        poss = fga - oreb + tov + 0.4 * fta
        opp_poss = val(away_pref, "FGA") - opp_oreb + val(away_pref, "TO") + 0.4 * val(away_pref, "FTA")
        poss_est = 0.5 * (poss + opp_poss)

        prefix = f"{home_pref}_"
        g[f"{prefix}POSS"] = poss_est
        g[f"{prefix}OFF_RTG"] = 100 * safe_div(pts, poss_est)
        g[f"{prefix}DEF_RTG"] = 100 * safe_div(opp_pts, poss_est)
        g[f"{prefix}NET_RTG"] = g[f"{prefix}OFF_RTG"] - g[f"{prefix}DEF_RTG"]
        g[f"{prefix}PACE"] = 48 * safe_div(poss_est, val(home_pref, "MIN") / 5)
        g[f"{prefix}EFG"] = safe_div(fgm + 0.5 * fg3m, fga)
        g[f"{prefix}TS"] = safe_div(pts, 2 * (fga + 0.44 * fta))
        g[f"{prefix}THREE_PAR"] = safe_div(fg3a, fga)
        g[f"{prefix}FTR"] = safe_div(fta, fga)
        g[f"{prefix}OREB_PCT"] = safe_div(oreb, oreb + opp_dreb)
        g[f"{prefix}DREB_PCT"] = safe_div(dreb, dreb + opp_oreb)
        g[f"{prefix}REB_PCT"] = safe_div(reb, reb + val(away_pref, "REB"))
        g[f"{prefix}AST_TOV"] = safe_div(ast, tov)
        g[f"{prefix}TOV_PCT"] = safe_div(tov, poss_est)

    return g


# ============================================================
#               STRING ENCODING + COMMENT ENCODING
# ============================================================

def encode_strings(df):
    df = df.copy()

    # Meaning-aware COMMENT encoding
    if "HOME_COMMENT" in df.columns:
        df["HOME_COMMENT_ENC"] = df["HOME_COMMENT"].apply(encode_comment)
    if "AWAY_COMMENT" in df.columns:
        df["AWAY_COMMENT_ENC"] = df["AWAY_COMMENT"].apply(encode_comment)

    # Encode all other textual columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes

    return df


# ============================================================
#                     MASTER PIPELINE
# ============================================================

def run_pipeline(box_path, players_path, output_path):

    print("Loading raw data...")
    box, players = load_data(box_path, players_path)

    print("Aggregating player → team stats...")
    team_stats = aggregate_player_to_team(players)

    print("Merging team stats with team boxscores...")
    df_team = merge_team_context(box, team_stats)

    print("Pivoting to single-row-per-game (HOME vs AWAY)...")
    games = pivot_to_games(df_team)

    print("Building player-level wide block...")
    player_block = build_player_block(players)
    # Use HOME_GAME_ID as the canonical GAME_ID for the merged frame.
    games["GAME_ID"] = games["HOME_GAME_ID"]
    games = games.merge(player_block, on="GAME_ID", how="left")

    print("Adding advanced metrics...")
    games = add_advanced_metrics(games)

    print("Computing game-level engineered features...")
    games = compute_game_level_features(games)

    print("Encoding COMMENT + strings...")
    games = encode_strings(games)

    print("Saving final dataset...")
    games.to_csv(output_path, index=False)

    print(f"✔ FINAL GAME-LEVEL DATA SAVED → {output_path}")
    print("Final shape:", games.shape)


# ============================================================
#                     EXECUTE SCRIPT
# ============================================================

if __name__ == "__main__":
    run_pipeline(
        box_path="/Users/mohamed/Desktop/Projects/NBA_Final/data/raw/boxscores_partial.csv",
        players_path="/Users/mohamed/Desktop/Projects/NBA_Final/data/raw/boxscores_players_partial.csv",
        output_path="/Users/mohamed/Desktop/Projects/NBA_Final/data/processed/data.csv"
    )
