import pandas as pd
import joblib


TRAIN_PATH = "data/processed/train.csv"
MODEL_PATH = "models/xgboost_dev_model.joblib"


def calculate_cost(y_true, predictions, false_negative_cost, false_positive_cost):
    false_negatives = ((y_true == 1) & (predictions == 0)).sum()
    false_positives = ((y_true == 0) & (predictions == 1)).sum()

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

    # Scenario:
    # Missing one fraud is assumed to be 10x more costly
    # than incorrectly flagging one legitimate transaction.
    false_negative_cost = 10
    false_positive_cost = 1

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        false_negatives, false_positives, total_cost = calculate_cost(
            y_validation,
            predictions,
            false_negative_cost,
            false_positive_cost,
        )

        results.append(
            {
                "Threshold": threshold,
                "False Negatives": false_negatives,
                "False Positives": false_positives,
                "Total Cost": total_cost,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print("COST-SENSITIVE THRESHOLD ANALYSIS")
    print("=" * 75)

    print(
        f"\nAssumed false-negative cost: {false_negative_cost}"
    )
    print(
        f"Assumed false-positive cost: {false_positive_cost}"
    )

    print(
        "\n"
        + results_df.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
            },
        )
    )

    best_row = results_df.loc[
        results_df["Total Cost"].idxmin()
    ]

    print("\nBest threshold under this cost assumption:")
    print(f"Threshold:       {best_row['Threshold']:.2f}")
    print(f"False Negatives:  {best_row['False Negatives']}")
    print(f"False Positives:  {best_row['False Positives']}")
    print(f"Total Cost:       {best_row['Total Cost']}")


if __name__ == "__main__":
    main()