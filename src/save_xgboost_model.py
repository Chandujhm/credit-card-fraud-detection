import pandas as pd
import joblib
from pathlib import Path

from xgboost import XGBClassifier


TRAIN_PATH = "data/processed/train.csv"
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "xgboost_fraud_model.joblib"


def main():
    print("Loading training data...")

    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

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

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved successfully:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()