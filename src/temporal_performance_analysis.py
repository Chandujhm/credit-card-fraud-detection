import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/xgboost_fraud_model.joblib"

THRESHOLD = 0.50


def main():
    print("Loading temporal test set...")

    test_df = pd.read_csv(TEST_PATH)

    print("Loading XGBoost model...")
    model = joblib.load(MODEL_PATH)

    # Divide the future test period into chronological slices.
    test_df = test_df.sort_values("Time").reset_index(drop=True)

    test_df["Time_Bin"] = pd.qcut(
        test_df["Time"],
        q=5,
        duplicates="drop",
    )

    results = []

    for time_bin, group in test_df.groupby(
        "Time_Bin",
        observed=True,
    ):
        X = group.drop(
            columns=["Class", "Time_Bin"]
        )
        y = group["Class"]

        probabilities = model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        fraud_count = int(y.sum())

        # ROC-AUC and PR-AUC require both classes.
        if y.nunique() == 2:
            pr_auc = average_precision_score(
                y,
                probabilities,
            )

            roc_auc = roc_auc_score(
                y,
                probabilities,
            )
        else:
            pr_auc = float("nan")
            roc_auc = float("nan")

        results.append(
            {
                "Time Period": str(time_bin),
                "Transactions": len(group),
                "Fraud Cases": fraud_count,
                "Fraud Rate": fraud_count / len(group),
                "Precision": precision_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "PR-AUC": pr_auc,
                "ROC-AUC": roc_auc,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print("TEMPORAL PERFORMANCE ANALYSIS")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Fraud Rate": "{:.4%}".format,
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