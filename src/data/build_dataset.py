import pandas as pd

from ..utils.logging_config import setup_logging
from ..utils import paths

logger = setup_logging(__name__)


def build_dataset():
    # 1. Load schedule and boxscores
    if not paths.SCHEDULE_CSV.exists():
        logger.error(f"Schedule CSV not found at {paths.SCHEDULE_CSV}")
        return

    if not paths.BOXSCORES_CSV.exists():
        logger.error(f"Boxscores CSV not found at {paths.BOXSCORES_CSV}")
        return
    if not paths.BOXSCORES_PLAYERS_CSV.exists():
        logger.error(f"Player boxscores CSV not found at {paths.BOXSCORES_PLAYERS_CSV}")
        return

    sched = pd.read_csv(paths.SCHEDULE_CSV, dtype={"GAME_ID": str})
    box = pd.read_csv(paths.BOXSCORES_CSV, dtype={"GAME_ID": str, "TEAM_ID": str})
    players = pd.read_csv(
        paths.BOXSCORES_PLAYERS_CSV,
        dtype={
            "GAME_ID": str,
            "TEAM_ID": str,
            "PLAYER_ID": str,
            "PLAYER_NAME": str,
            "START_POSITION": str,
            "COMMENT": str,
        },
    )

    required_sched_cols = {"GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"}
    if not required_sched_cols.issubset(sched.columns):
        logger.error(
            f"Schedule CSV missing required columns: {required_sched_cols - set(sched.columns)}"
        )
        return

    required_box_cols = {"GAME_ID", "TEAM_ID", "PTS", "REB", "AST", "STL", "BLK", "TO", "PF"}
    missing_box = required_box_cols - set(box.columns)
    if missing_box:
        logger.error(f"Boxscores CSV missing required columns: {missing_box}")
        return
    required_player_cols = {"GAME_ID", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "PTS", "REB", "AST", "STL", "BLK", "TO", "PF", "MIN"}
    missing_player = required_player_cols - set(players.columns)
    if missing_player:
        logger.error(f"Player boxscores CSV missing required columns: {missing_player}")
        return

    logger.info(f"Loaded schedule: {len(sched)} rows, boxscores: {len(box)} rows")

    # 2. Mark home vs away per team row
    merged = box.merge(
        sched[["GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"]],
        on="GAME_ID",
        how="inner",
        validate="many_to_one",
    )

    merged["IS_HOME"] = merged["TEAM_ID"] == merged["HOME_TEAM_ID"].astype(str)
    merged["IS_AWAY"] = merged["TEAM_ID"] == merged["AWAY_TEAM_ID"].astype(str)

    # Filter only rows that match either home or away team
    merged = merged[merged["IS_HOME"] | merged["IS_AWAY"]].copy()

    logger.info(f"After aligning with HOME/AWAY teams: {len(merged)} rows")

    # 3. Split into home and away frames
    home = merged[merged["IS_HOME"]].copy()
    away = merged[merged["IS_AWAY"]].copy()

    # Sanity: there should be exactly one home and one away row per GAME_ID
    dup_home = home["GAME_ID"].value_counts().gt(1).sum()
    dup_away = away["GAME_ID"].value_counts().gt(1).sum()
    if dup_home or dup_away:
        logger.warning(
            f"Found {dup_home} games with >1 home row and {dup_away} games with >1 away row"
        )

    # Keep only the columns we need from each side
    stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TO", "PF"]
    home = home[["GAME_ID"] + stat_cols].rename(
        columns={c: f"HOME_{c}" for c in stat_cols}
    )
    away = away[["GAME_ID"] + stat_cols].rename(
        columns={c: f"AWAY_{c}" for c in stat_cols}
    )

    # 4. Merge home and away into a single game-level row
    games = (
        home.merge(away, on="GAME_ID", how="inner", validate="one_to_one")
        .merge(
            sched[
                [
                    "GAME_ID",
                    "GAME_DATE",
                    "SEASON" if "SEASON" in sched.columns else "SEASON_ID",
                    "HOME_TEAM_ID",
                    "AWAY_TEAM_ID",
                ]
            ],
            on="GAME_ID",
            how="left",
        )
        .drop_duplicates(subset=["GAME_ID"])
    )

    # 5. Compute target: home margin (home points - away points)
    games["HOME_MARGIN"] = games["HOME_PTS"] - games["AWAY_PTS"]

    # Also provide a binary label if you ever want it
    games["HOME_TEAM_WON"] = (games["HOME_MARGIN"] > 0).astype(int)

    # 6. Advanced team-level metrics per game
    def _safe_div(n, d):
        return n / d if d not in (0, None) else 0

    def _poss(h_fga, h_fta, h_fgm, h_oreb, h_to, a_fga, a_fta, a_fgm, a_oreb, a_to, a_dreb, h_dreb):
        # Standard estimate of possessions combining both teams to reduce noise
        h_part = h_fga + 0.4 * h_fta - 1.07 * _safe_div(h_oreb, h_oreb + a_dreb) * (h_fga - h_fgm) + h_to
        a_part = a_fga + 0.4 * a_fta - 1.07 * _safe_div(a_oreb, a_oreb + h_dreb) * (a_fga - a_fgm) + a_to
        return 0.5 * (h_part + a_part)

    # Compute possessions per game
    games["HOME_POSSESSIONS"] = games.apply(
        lambda r: _poss(
            r["HOME_FGA"], r["HOME_FTA"], r["HOME_FGM"], r["HOME_OREB"], r["HOME_TO"],
            r["AWAY_FGA"], r["AWAY_FTA"], r["AWAY_FGM"], r["AWAY_OREB"], r["AWAY_TO"],
            r["AWAY_DREB"], r["HOME_DREB"]
        ),
        axis=1,
    )
    games["AWAY_POSSESSIONS"] = games.apply(
        lambda r: _poss(
            r["AWAY_FGA"], r["AWAY_FTA"], r["AWAY_FGM"], r["AWAY_OREB"], r["AWAY_TO"],
            r["HOME_FGA"], r["HOME_FTA"], r["HOME_FGM"], r["HOME_OREB"], r["HOME_TO"],
            r["HOME_DREB"], r["AWAY_DREB"]
        ),
        axis=1,
    )

    # Ratings
    games["HOME_OFF_RTG"] = games.apply(lambda r: _safe_div(r["HOME_PTS"], r["HOME_POSSESSIONS"]) * 100, axis=1)
    games["HOME_DEF_RTG"] = games.apply(lambda r: _safe_div(r["AWAY_PTS"], r["HOME_POSSESSIONS"]) * 100, axis=1)
    games["HOME_NET_RTG"] = games["HOME_OFF_RTG"] - games["HOME_DEF_RTG"]

    games["AWAY_OFF_RTG"] = games.apply(lambda r: _safe_div(r["AWAY_PTS"], r["AWAY_POSSESSIONS"]) * 100, axis=1)
    games["AWAY_DEF_RTG"] = games.apply(lambda r: _safe_div(r["HOME_PTS"], r["AWAY_POSSESSIONS"]) * 100, axis=1)
    games["AWAY_NET_RTG"] = games["AWAY_OFF_RTG"] - games["AWAY_DEF_RTG"]

    # Pace (per 48 minutes, using team minutes / 5 players)
    games["PACE"] = games.apply(
        lambda r: _safe_div(48 * (r["HOME_POSSESSIONS"] + r["AWAY_POSSESSIONS"]) / 2, _safe_div(r["HOME_MIN"], 5)),
        axis=1,
    )

    # Shooting efficiency
    for side in ["HOME", "AWAY"]:
        fgm = games[f"{side}_FGM"]
        fga = games[f"{side}_FGA"]
        fg3m = games[f"{side}_FG3M"]
        fg3a = games[f"{side}_FG3A"]
        fta = games[f"{side}_FTA"]
        ftm = games[f"{side}_FTM"]
        pts = games[f"{side}_PTS"]

        games[f"{side}_EFG"] = (fgm + 0.5 * fg3m) / fga.replace(0, pd.NA)
        games[f"{side}_TS"] = pts / (2 * (fga + 0.44 * fta).replace(0, pd.NA))
        games[f"{side}_3PAR"] = fg3a / fga.replace(0, pd.NA)
        games[f"{side}_FTR"] = fta / fga.replace(0, pd.NA)
        games[f"{side}_FT_PER_FGA"] = ftm / fga.replace(0, pd.NA)

    # Rebounding percentages
    games["HOME_OREB_PCT"] = games.apply(lambda r: _safe_div(r["HOME_OREB"], r["HOME_OREB"] + r["AWAY_DREB"]), axis=1)
    games["AWAY_OREB_PCT"] = games.apply(lambda r: _safe_div(r["AWAY_OREB"], r["AWAY_OREB"] + r["HOME_DREB"]), axis=1)
    games["HOME_DREB_PCT"] = games.apply(lambda r: _safe_div(r["HOME_DREB"], r["HOME_DREB"] + r["AWAY_OREB"]), axis=1)
    games["AWAY_DREB_PCT"] = games.apply(lambda r: _safe_div(r["AWAY_DREB"], r["AWAY_DREB"] + r["HOME_OREB"]), axis=1)
    games["HOME_REB_PCT"] = games.apply(lambda r: _safe_div(r["HOME_REB"], r["HOME_REB"] + r["AWAY_REB"]), axis=1)
    games["AWAY_REB_PCT"] = games.apply(lambda r: _safe_div(r["AWAY_REB"], r["HOME_REB"] + r["AWAY_REB"]), axis=1)

    # Playmaking / turnovers
    for side in ["HOME", "AWAY"]:
        games[f"{side}_AST_PCT"] = games.apply(lambda r: _safe_div(r[f"{side}_AST"], r[f"{side}_FGM"]), axis=1)
        games[f"{side}_AST_TOV"] = games.apply(lambda r: _safe_div(r[f"{side}_AST"], r[f"{side}_TO"]), axis=1)
        games[f"{side}_AST_RATIO"] = games.apply(
            lambda r: _safe_div(r[f"{side}_AST"], r[f"{side}_POSSESSIONS"]) * 100, axis=1
        )
        games[f"{side}_TO_PCT"] = games.apply(
            lambda r: _safe_div(r[f"{side}_TO"], r[f"{side}_POSSESSIONS"]) * 100, axis=1
        )

    # PIE approximation
    def _pie(row, side):
        opp = "AWAY" if side == "HOME" else "HOME"
        num = (
            row[f"{side}_PTS"] + row[f"{side}_FGM"] + row[f"{side}_FTM"] - row[f"{side}_FTA"]
            + row[f"{side}_OREB"] + row[f"{side}_AST"] + row[f"{side}_STL"] + row[f"{side}_BLK"] - row[f"{side}_TO"]
        )
        den = num + (
            row[f"{opp}_PTS"] + row[f"{opp}_FGM"] + row[f"{opp}_FTM"] - row[f"{opp}_FTA"]
            + row[f"{opp}_OREB"] + row[f"{opp}_AST"] + row[f"{opp}_STL"] + row[f"{opp}_BLK"] - row[f"{opp}_TO"]
        )
        return _safe_div(num, den)

    games["HOME_PIE"] = games.apply(lambda r: _pie(r, "HOME"), axis=1)
    games["AWAY_PIE"] = games.apply(lambda r: _pie(r, "AWAY"), axis=1)

    # Differentials (home - away) for useful model features
    diff_pairs = [
        "OFF_RTG",
        "DEF_RTG",
        "NET_RTG",
        "EFG",
        "TS",
        "3PAR",
        "FTR",
        "FT_PER_FGA",
        "OREB_PCT",
        "DREB_PCT",
        "REB_PCT",
        "AST_PCT",
        "AST_TOV",
        "AST_RATIO",
        "TO_PCT",
        "PIE",
    ]
    for metric in diff_pairs:
        games[f"{metric}_DIFF"] = games[f"HOME_{metric}"] - games[f"AWAY_{metric}"]

    logger.info(
        f"Built game-level dataset with {len(games)} rows. "
        f"Mean HOME_MARGIN={games['HOME_MARGIN'].mean():.2f}"
    )

    # 7. Save to processed
    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = paths.PROCESSED_DIR / "dataset.csv"
    games.to_csv(out_path, index=False)
    logger.info(f"Saved dataset to {out_path.resolve()}")

    # 8. Build combined player+team advanced view (one row per player per game)
    team_rows = []
    base_stats = [
        "PTS",
        "REB",
        "OREB",
        "DREB",
        "AST",
        "STL",
        "BLK",
        "TO",
        "PF",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "MIN",
        "POSSESSIONS",
        "OFF_RTG",
        "DEF_RTG",
        "NET_RTG",
        "EFG",
        "TS",
        "3PAR",
        "FTR",
        "FT_PER_FGA",
        "OREB_PCT",
        "DREB_PCT",
        "REB_PCT",
        "AST_PCT",
        "AST_TOV",
        "AST_RATIO",
        "TO_PCT",
        "PIE",
    ]

    for side in ["HOME", "AWAY"]:
        opp = "AWAY" if side == "HOME" else "HOME"
        row = pd.DataFrame({
            "GAME_ID": games["GAME_ID"],
            "GAME_DATE": games["GAME_DATE"],
            "TEAM_ID": games[f"{side}_TEAM_ID"].astype(str),
            "OPP_TEAM_ID": games[f"{opp}_TEAM_ID"].astype(str),
            "IS_HOME": side == "HOME",
            "PACE": games["PACE"],
        })
        for col in base_stats:
            row[f"{col}"] = games[f"{side}_{col}"]
        team_rows.append(row)

    team_df = pd.concat(team_rows, ignore_index=True)

    combined = players.merge(team_df, on=["GAME_ID", "TEAM_ID"], how="left", validate="many_to_one")
    combined_out = paths.PROCESSED_DIR / "player_team_advanced.csv"
    combined.to_csv(combined_out, index=False)
    logger.info(f"Saved combined player + team advanced dataset to {combined_out.resolve()}")


def main():
    logger.info("=== Building enriched dataset (game-level) ===")
    build_dataset()


if __name__ == "__main__":
    main()
