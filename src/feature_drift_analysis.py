import pandas as pd
from scipy.stats import ks_2samp


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


def main():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)

    print("Loading temporal test data...")
    test_df = pd.read_csv(TEST_PATH)

    # Time is intentionally different because the test set
    # represents a future time period. We therefore exclude it
    # from predictive feature drift analysis.
    feature_columns = [
        column
        for column in train_df.columns
        if column not in ["Class", "Time"]
    ]

    results = []

    for feature in feature_columns:
        train_values = train_df[feature].dropna()
        test_values = test_df[feature].dropna()

        ks_statistic, p_value = ks_2samp(
            train_values,
            test_values,
        )

        train_median = train_values.median()
        test_median = test_values.median()

        if ks_statistic >= 0.20:
            drift_level = "High"
        elif ks_statistic >= 0.10:
            drift_level = "Moderate"
        elif ks_statistic >= 0.05:
            drift_level = "Low"
        else:
            drift_level = "Minimal"

        results.append(
            {
                "Feature": feature,
                "KS Statistic": ks_statistic,
                "P-Value": p_value,
                "Train Median": train_median,
                "Test Median": test_median,
                "Drift Level": drift_level,
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "KS Statistic",
        ascending=False,
    )

    print("\n" + "=" * 100)
    print("PREDICTIVE FEATURE DRIFT ANALYSIS")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "KS Statistic": "{:.4f}".format,
                "P-Value": "{:.6f}".format,
                "Train Median": "{:.4f}".format,
                "Test Median": "{:.4f}".format,
            },
        )
    )

    print("\n" + "=" * 100)
    print("FEATURES REQUIRING MONITORING")
    print("=" * 100)

    monitoring_features = results_df[
        results_df["KS Statistic"] >= 0.10
    ]

    if monitoring_features.empty:
        print("No features exceeded the monitoring threshold.")
    else:
        print(
            monitoring_features[
                [
                    "Feature",
                    "KS Statistic",
                    "P-Value",
                    "Drift Level",
                ]
            ].to_string(
                index=False,
                formatters={
                    "KS Statistic": "{:.4f}".format,
                    "P-Value": "{:.6f}".format,
                },
            )
        )


if __name__ == "__main__":
    main()