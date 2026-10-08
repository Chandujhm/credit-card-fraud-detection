import pandas as pd
import joblib
from pathlib import Path

from xgboost import XGBClassifier


TRAIN_PATH = "data/processed/train.csv"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgboost_dev_model.joblib"


def main():
    print("Loading training data...")

    train_df = pd.read_csv(TRAIN_PATH)

    # Chronological development/validation split
    split_index = int(len(train_df) * 0.80)

    dev_df = train_df.iloc[:split_index].copy()
    validation_df = train_df.iloc[split_index:].copy()

    X_dev = dev_df.drop(columns=["Class"])
    y_dev = dev_df["Class"]

    print(f"Development shape: {X_dev.shape}")
    print(f"Validation shape:  {validation_df.shape}")

    negative_count = (y_dev == 0).sum()
    positive_count = (y_dev == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print(f"\nDevelopment legitimate transactions: {negative_count}")
    print(f"Development fraud transactions:      {positive_count}")
    print(f"Scale positive weight:               {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost only on development data...")
    model.fit(X_dev, y_dev)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\nDevelopment-only model saved successfully:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()