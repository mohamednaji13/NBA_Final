from pathlib import Path

import pandas as pd


def split_dataset(
    input_path: Path,
    output_dir: Path,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Split a CSV into train/val/test subsets with the given fractions.
    Fractions must sum to <= 1; remaining rows go to test.
    """
    df = pd.read_csv(input_path)
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = shuffled.iloc[:train_end]
    val_df = shuffled.iloc[train_end:val_end]
    test_df = shuffled.iloc[val_end:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Saved: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "data" / "processed" / "data.csv"
    output_dir = root / "data" / "processed"

    split_dataset(input_path=input_path, output_dir=output_dir, train_frac=0.7, val_frac=0.2, seed=42)


if __name__ == "__main__":
    main()
