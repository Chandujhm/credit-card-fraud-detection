import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/xgboost_fraud_model.joblib"

SELECTED_THRESHOLD = 0.50


def main():
    print("Loading untouched temporal test set...")
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    print(f"Test shape: {X_test.shape}")
    print(f"Fraud cases: {y_test.sum()}")

    print("\nLoading final XGBoost model...")
    model = joblib.load(MODEL_PATH)

    # Generate fraud probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Apply the threshold selected using validation data
    predictions = (
        probabilities >= SELECTED_THRESHOLD
    ).astype(int)

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

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 70)
    print("FINAL TEMPORAL TEST EVALUATION")
    print("=" * 70)

    print(f"\nLocked threshold: {SELECTED_THRESHOLD:.2f}")

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

    print("\nFinal Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")


if __name__ == "__main__":
    main()