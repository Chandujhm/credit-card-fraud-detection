import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
XGB_MODEL_PATH = "models/xgboost_fraud_model.joblib"


def evaluate_model(name, y_true, predictions, probabilities):
    return {
        "Model": name,
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y_true, probabilities
        ),
        "ROC-AUC": roc_auc_score(
            y_true, probabilities
        ),
    }


def main():
    print("Loading data...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    results = []

    # ---------------------------------------------------------
    # 1. Naive baseline
    # ---------------------------------------------------------
    naive_predictions = [0] * len(y_test)
    naive_probabilities = [0.0] * len(y_test)

    results.append(
        evaluate_model(
            "Naive Baseline",
            y_test,
            naive_predictions,
            naive_probabilities,
        )
    )

    # ---------------------------------------------------------
    # 2. Logistic Regression
    # ---------------------------------------------------------
    print("Training Logistic Regression...")

    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)

    logistic_probabilities = logistic_model.predict_proba(
        X_test
    )[:, 1]

    logistic_predictions = (
        logistic_probabilities >= 0.50
    ).astype(int)

    results.append(
        evaluate_model(
            "Logistic Regression",
            y_test,
            logistic_predictions,
            logistic_probabilities,
        )
    )

    # ---------------------------------------------------------
    # 3. XGBoost
    # ---------------------------------------------------------
    print("Loading saved XGBoost model...")

    xgb_model = joblib.load(XGB_MODEL_PATH)

    xgb_probabilities = xgb_model.predict_proba(
        X_test
    )[:, 1]

    # Locked threshold selected from validation data
    xgb_predictions = (
        xgb_probabilities >= 0.50
    ).astype(int)

    results.append(
        evaluate_model(
            "XGBoost",
            y_test,
            xgb_predictions,
            xgb_probabilities,
        )
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON — TEMPORAL TEST SET")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
                "PR-AUC": "{:.4f}".format,
                "ROC-AUC": "{:.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()