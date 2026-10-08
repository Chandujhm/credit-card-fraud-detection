import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


def main():
    print("Loading training and test data...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    feature_columns = [
        column
        for column in train_df.columns
        if column != "Class"
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["Class"]

    X_test = test_df[feature_columns]
    y_test = test_df["Class"]

    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression baseline...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION BASELINE RESULTS")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(f"\nPR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()