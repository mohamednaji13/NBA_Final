from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "data.csv"


def update_processed_manual(source_csv: Path) -> None:
    """
    Manually append/overwrite processed data from a given CSV into data/processed/data.csv.
    Use this after you’ve run your merge/normalize pipeline for the latest games.
    """
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    df = pd.read_csv(source_csv)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Updated {PROCESSED_PATH} with {len(df)} rows from {source_csv}")


def main():
    # Replace this path with your freshly generated processed file for the day.
    # Example: Path(\"data/processed/data_new.csv\")
    source = Path("data/processed/data.csv")
    update_processed_manual(source)


if __name__ == "__main__":
    main()
