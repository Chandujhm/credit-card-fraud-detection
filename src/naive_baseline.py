import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

TEST_PATH = "data/processed/test.csv"


def main():
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)

    y_test = test_df["Class"]

    # Naive strategy: classify every transaction as legitimate
    predictions = [0] * len(y_test)

    # Probability assigned to fraud is always 0
    probabilities = [0.0] * len(y_test)

    print("\n" + "=" * 60)
    print("NAIVE ALL-LEGITIMATE BASELINE")
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