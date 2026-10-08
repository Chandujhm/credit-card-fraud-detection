import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path


TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/xgboost_fraud_model.joblib"
REPORT_DIR = Path("reports/shap")


def main():
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(columns=["Class"])

    print("Loading XGBoost model...")
    model = joblib.load(MODEL_PATH)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Use a representative sample for SHAP.
    # The full test set is unnecessarily expensive for explanation.
    sample_size = min(5000, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42,
    )

    print(f"Generating SHAP explanations for {sample_size} transactions...")

    explainer = shap.Explainer(
    model.predict,
    X_sample,
)

    shap_values = explainer(X_sample)

    # ---------------------------------------------------------
    # Global feature importance
    # ---------------------------------------------------------
    print("Creating global SHAP summary plot...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
    )

    plt.tight_layout()

    summary_path = REPORT_DIR / "shap_summary.png"

    plt.savefig(
        summary_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {summary_path}")

    # ---------------------------------------------------------
    # Mean absolute SHAP importance
    # ---------------------------------------------------------
    importance = pd.DataFrame(
        {
            "Feature": X_sample.columns,
            "Mean Absolute SHAP": abs(shap_values.values).mean(axis=0),
        }
    ).sort_values(
        "Mean Absolute SHAP",
        ascending=False,
    )

    importance_path = REPORT_DIR / "shap_feature_importance.csv"

    importance.to_csv(
        importance_path,
        index=False,
    )

    print(f"Saved: {importance_path}")

    print("\n" + "=" * 70)
    print("TOP 15 FEATURES BY SHAP IMPORTANCE")
    print("=" * 70)

    print(
        importance.head(15).to_string(
            index=False,
            formatters={
                "Mean Absolute SHAP": "{:.6f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()