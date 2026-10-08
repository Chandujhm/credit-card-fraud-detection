import pandas as pd
import joblib

from sklearn.metrics import precision_score, recall_score, f1_score


TRAIN_PATH = "data/processed/train.csv"
MODEL_PATH = "models/xgboost_fraud_model.joblib"


def main():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)

    # Keep the final temporal test set completely untouched.
    # We create a development/validation split from the training period.
    split_index = int(len(train_df) * 0.80)

    dev_df = train_df.iloc[:split_index].copy()
    validation_df = train_df.iloc[split_index:].copy()

    X_validation = validation_df.drop(columns=["Class"])
    y_validation = validation_df["Class"]

    print(f"Development rows: {len(dev_df)}")
    print(f"Validation rows:  {len(validation_df)}")

    print("\nLoading saved XGBoost model...")
    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X_validation)[:, 1]

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("XGBOOST DEVELOPMENT THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
            },
        )
    )

    best_row = results_df.loc[
        results_df["F1"].idxmax()
    ]

    print("\nBest development threshold by F1:")
    print(f"Threshold: {best_row['Threshold']:.2f}")
    print(f"Precision: {best_row['Precision']:.4f}")
    print(f"Recall:    {best_row['Recall']:.4f}")
    print(f"F1:        {best_row['F1']:.4f}")


if __name__ == "__main__":
    main()