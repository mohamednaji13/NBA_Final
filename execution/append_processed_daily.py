import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "processed" / "data.csv"


def append_new_rows(source_csv: Path, base_csv: Path = BASE_PATH) -> None:
    """Append only truly new GAME_IDs from source into base_csv."""
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    new_df = pd.read_csv(source_csv, dtype={"GAME_ID": str})
    if "GAME_ID" not in new_df.columns:
        raise ValueError("Source CSV must contain GAME_ID")

    if base_csv.exists():
        base_df = pd.read_csv(base_csv, dtype={"GAME_ID": str})
    else:
        base_df = pd.DataFrame(columns=new_df.columns)

    before = len(base_df)
    combined = pd.concat([base_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["GAME_ID"])
    added = len(combined) - before

    combined.to_csv(base_csv, index=False)
    print(f"Appended {added} new rows into {base_csv}. Total rows now: {len(combined)}")


def main():
    parser = argparse.ArgumentParser(description="Append new processed rows into data.csv")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to CSV containing only new processed games (with GAME_ID column).",
    )
    args = parser.parse_args()
    append_new_rows(args.source)


if __name__ == "__main__":
    main()
