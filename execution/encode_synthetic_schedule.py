from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SYN_PATH = ROOT / "execution" / "nba_2025_26_synthetic_schedule.csv"
REAL_SCHEDULE = ROOT / "data" / "raw" / "games_schedule.csv"


def build_team_mapping(schedule_path: Path) -> dict:
    """Build TEAM_ABBREVIATION -> TEAM_ID mapping from an existing schedule file."""
    df = pd.read_csv(schedule_path)
    cols = []
    for col_set in [
        ("HOME_TEAM_ABBREVIATION", "HOME_TEAM_ID"),
        ("AWAY_TEAM_ABBREVIATION", "AWAY_TEAM_ID"),
        ("TEAM_ABBREVIATION", "TEAM_ID"),
    ]:
        if col_set[0] in df.columns and col_set[1] in df.columns:
            cols.append(col_set)

    mapping = {}
    for abbr_col, id_col in cols:
        mapping.update(
            df[[abbr_col, id_col]]
            .dropna()
            .drop_duplicates()
            .set_index(abbr_col)[id_col]
            .to_dict()
        )
    return mapping


def season_id_from_label(season_label: str) -> int:
    """
    Convert season label like '2025-26' to SEASON_ID (e.g., 22025).
    Mirrors NBA season id pattern used elsewhere in the project.
    """
    start_year = int(str(season_label).split("-")[0])
    return 22000 + (start_year % 10000)


def encode_synthetic_schedule():
    mapping = build_team_mapping(REAL_SCHEDULE)
    if not mapping:
        raise RuntimeError(f"Could not build team mapping from {REAL_SCHEDULE}")

    df = pd.read_csv(SYN_PATH, dtype={"GAME_ID": str})
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df.rename(columns={"HOME_TEAM": "HOME_TEAM_ABBREVIATION", "AWAY_TEAM": "AWAY_TEAM_ABBREVIATION"})

    df["HOME_TEAM_ID"] = df["HOME_TEAM_ABBREVIATION"].map(mapping).fillna(-1).astype(int)
    df["AWAY_TEAM_ID"] = df["AWAY_TEAM_ABBREVIATION"].map(mapping).fillna(-1).astype(int)

    if "SEASON_ID" not in df.columns:
        df["SEASON_ID"] = df["SEASON"].apply(season_id_from_label)

    # Build MATCHUP consistent with existing schedules (away @ home)
    df["MATCHUP"] = df["AWAY_TEAM_ABBREVIATION"] + " @ " + df["HOME_TEAM_ABBREVIATION"]

    df.to_csv(SYN_PATH, index=False)
    print(f"Encoded synthetic schedule saved to {SYN_PATH.resolve()}")


if __name__ == "__main__":
    encode_synthetic_schedule()
