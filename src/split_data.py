import pandas as pd
from pathlib import Path

print("split_data.py is running")
RAW_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")


def main():
    df = pd.read_csv(RAW_PATH)

    # Sort chronologically
    df = df.sort_values("Time").reset_index(drop=True)

    # 80/20 chronological split
    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("=" * 60)
    print("TEMPORAL TRAIN/TEST SPLIT")
    print("=" * 60)

    print(f"\nTotal records: {len(df):,}")
    print(f"Training records: {len(train_df):,}")
    print(f"Test records: {len(test_df):,}")

    print("\nTraining time range:")
    print(
        f"{train_df['Time'].min():.2f} "
        f"-> {train_df['Time'].max():.2f}"
    )

    print("\nTest time range:")
    print(
        f"{test_df['Time'].min():.2f} "
        f"-> {test_df['Time'].max():.2f}"
    )

    print("\nTraining class distribution:")
    print(train_df["Class"].value_counts())

    print("\nTest class distribution:")
    print(test_df["Class"].value_counts())

    print("\nFiles saved:")
    print(train_path)
    print(test_path)


if __name__ == "__main__":
    main()