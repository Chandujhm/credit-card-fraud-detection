import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/xgboost_fraud_model.joblib"


def main():
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    print("Loading saved XGBoost model...")
    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("SAVED XGBOOST MODEL EVALUATION")
    print("=" * 60)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print(f"\nPR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()