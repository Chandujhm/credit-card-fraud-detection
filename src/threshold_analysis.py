import pandas as pd
import joblib

from sklearn.metrics import precision_score, recall_score, f1_score


TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/xgboost_fraud_model.joblib"


def main():
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    print("Loading saved XGBoost model...")
    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
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
    print("XGBOOST THRESHOLD ANALYSIS")
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


if __name__ == "__main__":
    main()