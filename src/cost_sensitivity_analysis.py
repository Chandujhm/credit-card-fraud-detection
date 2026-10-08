import pandas as pd
import joblib


TRAIN_PATH = "data/processed/train.csv"
MODEL_PATH = "models/xgboost_dev_model.joblib"


def calculate_cost(
    y_true,
    predictions,
    false_negative_cost,
    false_positive_cost,
):
    false_negatives = (
        (y_true == 1) & (predictions == 0)
    ).sum()

    false_positives = (
        (y_true == 0) & (predictions == 1)
    ).sum()

    total_cost = (
        false_negatives * false_negative_cost
        + false_positives * false_positive_cost
    )

    return false_negatives, false_positives, total_cost


def main():
    print("Loading validation data...")

    train_df = pd.read_csv(TRAIN_PATH)

    # First 80% = development
    # Last 20% = unseen validation period
    split_index = int(len(train_df) * 0.80)

    validation_df = train_df.iloc[split_index:].copy()

    X_validation = validation_df.drop(columns=["Class"])
    y_validation = validation_df["Class"]

    print(f"Validation shape: {X_validation.shape}")
    print(f"Validation fraud cases: {y_validation.sum()}")

    print("\nLoading development-trained XGBoost model...")
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

    cost_ratios = [2, 5, 10, 20, 50]

    results = []

    for ratio in cost_ratios:

        false_negative_cost = ratio
        false_positive_cost = 1

        best_threshold = None
        best_cost = float("inf")
        best_fn = None
        best_fp = None

        for threshold in thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            false_negatives, false_positives, total_cost = (
                calculate_cost(
                    y_validation,
                    predictions,
                    false_negative_cost,
                    false_positive_cost,
                )
            )

            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
                best_fn = false_negatives
                best_fp = false_positives

        results.append(
            {
                "FN:FP Cost Ratio": f"{ratio}:1",
                "Best Threshold": best_threshold,
                "False Negatives": best_fn,
                "False Positives": best_fp,
                "Minimum Cost": best_cost,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("COST SENSITIVITY ANALYSIS")
    print("=" * 80)

    print(
        "\n"
        + results_df.to_string(
            index=False,
            formatters={
                "Best Threshold": "{:.2f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()