import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_fraud_model.joblib"

THRESHOLD = 0.50

FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fraud Detection Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* -----------------------------------------------------
           Professional motion system
           - Short 180–260ms transitions
           - Gentle entrance animation
           - No bounce, spin, glow, or excessive movement
        ----------------------------------------------------- */

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(6px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes softFade {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

        [data-testid="stAppViewContainer"] .main .block-container {
            animation: fadeUp 0.24s ease-out both;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 12px;
            padding: 16px;
            transition:
                transform 0.2s ease,
                border-color 0.2s ease,
                box-shadow 0.2s ease,
                background-color 0.2s ease;
            animation: softFade 0.22s ease-out both;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(128, 128, 128, 0.45);
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        }

        [data-testid="stButton"] > button,
        [data-testid="stFormSubmitButton"] > button,
        [data-testid="stDownloadButton"] > button {
            transition:
                transform 0.18s ease,
                box-shadow 0.18s ease,
                border-color 0.18s ease;
        }

        [data-testid="stButton"] > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover,
        [data-testid="stDownloadButton"] > button:hover {
            transform: translateY(-1px);
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            animation: fadeUp 0.24s ease-out both;
        }

        [data-testid="stAlert"] {
            animation: fadeUp 0.22s ease-out both;
        }

        [data-testid="stProgress"] {
            animation: softFade 0.24s ease-out both;
        }

        [data-testid="stExpander"] {
            transition:
                border-color 0.2s ease,
                background-color 0.2s ease;
        }

        [data-testid="stExpander"]:hover {
            border-color: rgba(128, 128, 128, 0.35);
        }

        .stCaption {
            opacity: 0.75;
            transition: opacity 0.2s ease;
        }

        .stCaption:hover {
            opacity: 0.95;
        }

        h1 {
            font-weight: 700;
            letter-spacing: -0.5px;
            animation: fadeUp 0.22s ease-out both;
        }

        h2,
        h3 {
            font-weight: 650;
        }

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Fraud Detection")
    st.caption("Model Monitoring & Analysis")

    st.divider()

    st.subheader("Model")
    st.write("XGBoost")

    st.subheader("Operating Threshold")
    st.metric("Threshold", "0.50")

    st.divider()

    if model_loaded:
        st.success("Model loaded")
    else:
        st.error("Model unavailable")

    st.caption("Temporal validation")
    st.caption("Future-period test evaluation")


# =========================================================
# HEADER
# =========================================================

st.title("Fraud Detection Intelligence")

st.caption(
    "Advanced Credit Card Fraud Detection • "
    "XGBoost • Temporal Validation"
)

st.divider()


# =========================================================
# KPI SUMMARY
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("PR-AUC", "0.7994")

with col2:
    st.metric("Precision", "87.69%")

with col3:
    st.metric("Recall", "76.00%")

with col4:
    st.metric("F1 Score", "0.8143")


# =========================================================
# MODEL OVERVIEW
# =========================================================

st.subheader("Model Overview")

st.write(
    """
    The system uses an XGBoost classifier to identify potentially
    fraudulent credit card transactions. The operating threshold
    was selected using a chronological validation period and the
    final performance was evaluated on a completely untouched
    future temporal test set.
    """
)


# =========================================================
# DECISION STRATEGY
# =========================================================

st.subheader("Decision Strategy")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Operating Threshold", "0.50")

with col2:
    st.metric("Fraud Detected", "57 / 75")

with col3:
    st.metric("False Positives", "8")

st.caption(
    "The 0.50 operating threshold was selected using a chronological "
    "validation period and then applied unchanged to the future test set."
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.subheader("Model Performance")

performance_data = {
    "Metric": [
        "Precision",
        "Recall",
        "F1 Score",
        "PR-AUC",
        "ROC-AUC",
    ],
    "Score": [
        "87.69%",
        "76.00%",
        "81.43%",
        "0.7994",
        "0.9864",
    ],
}

st.dataframe(
    performance_data,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.divider()

st.subheader("Confusion Matrix")

cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    st.metric("True Negatives", "56,879")
    st.metric("False Positives", "8")

with cm_col2:
    st.metric("False Negatives", "18")
    st.metric("True Positives", "57")

st.caption(
    "On the future test set, the model correctly identified 57 fraudulent "
    "transactions while generating only 8 false positive alerts. "
    "18 fraudulent transactions were missed."
)


# =========================================================
# TEMPORAL PERFORMANCE MONITORING
# =========================================================

st.divider()

st.subheader("Temporal Performance Monitoring")

st.write(
    """
    The future test set is divided into five chronological periods to
    examine whether model performance remains stable as transaction
    patterns change over time.
    """
)


temporal_data = pd.DataFrame(
    {
        "Period": [
            "Period 1",
            "Period 2",
            "Period 3",
            "Period 4",
            "Period 5",
        ],
        "Transactions": [
            11396,
            11390,
            11394,
            11389,
            11393,
        ],
        "Fraud Cases": [
            18,
            23,
            16,
            8,
            10,
        ],
        "Fraud Rate": [
            0.001580,
            0.002019,
            0.001404,
            0.000702,
            0.000878,
        ],
        "Precision": [
            0.9375,
            1.0000,
            1.0000,
            0.6000,
            0.6667,
        ],
        "Recall": [
            0.8333,
            0.7826,
            0.7500,
            0.7500,
            0.6000,
        ],
        "F1": [
            0.8824,
            0.8780,
            0.8571,
            0.6667,
            0.6316,
        ],
        "PR-AUC": [
            0.8979,
            0.8432,
            0.7732,
            0.7628,
            0.5929,
        ],
        "ROC-AUC": [
            0.9988,
            0.9916,
            0.9926,
            0.9767,
            0.9510,
        ],
    }
)


# =========================================================
# PERFORMANCE TREND
# =========================================================

st.markdown("#### Performance Trend")

trend_data = temporal_data.set_index("Period")[
    ["F1", "PR-AUC", "Precision", "Recall"]
]

st.line_chart(
    trend_data,
    y_label="Score",
    x_label="Future Test Period",
)


temporal_col1, temporal_col2, temporal_col3 = st.columns(3)

with temporal_col1:
    st.metric(
        "Period 1 F1",
        "88.24%",
    )

with temporal_col2:
    st.metric(
        "Period 5 F1",
        "63.16%",
    )

with temporal_col3:
    st.metric(
        "Period 5 PR-AUC",
        "0.5929",
    )


# =========================================================
# PERIOD-LEVEL RESULTS
# =========================================================

st.markdown("#### Period-Level Results")

display_temporal = temporal_data.copy()

display_temporal["Fraud Rate"] = (
    display_temporal["Fraud Rate"] * 100
).round(4).astype(str) + "%"

display_temporal["Precision"] = (
    display_temporal["Precision"] * 100
).round(2).astype(str) + "%"

display_temporal["Recall"] = (
    display_temporal["Recall"] * 100
).round(2).astype(str) + "%"

display_temporal["F1"] = (
    display_temporal["F1"] * 100
).round(2).astype(str) + "%"

display_temporal["PR-AUC"] = (
    display_temporal["PR-AUC"]
).round(4)

display_temporal["ROC-AUC"] = (
    display_temporal["ROC-AUC"]
).round(4)

st.dataframe(
    display_temporal,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# FRAUD RATE MONITORING
# =========================================================

st.markdown("#### Fraud Rate Monitoring")

st.write(
    """
    Observed fraud prevalence is monitored across chronological
    future-test periods. Changes in fraud prevalence can affect
    model performance metrics and should be considered when
    evaluating production stability.
    """
)


fraud_rate_chart_data = temporal_data.set_index("Period")[
    ["Fraud Rate"]
].copy()

fraud_rate_chart_data["Fraud Rate (%)"] = (
    fraud_rate_chart_data["Fraud Rate"] * 100
)

fraud_rate_chart_data = fraud_rate_chart_data[
    ["Fraud Rate (%)"]
]

st.line_chart(
    fraud_rate_chart_data,
    y_label="Fraud Rate (%)",
    x_label="Future Test Period",
)


fraud_col1, fraud_col2, fraud_col3 = st.columns(3)

with fraud_col1:
    st.metric(
        "Highest Fraud Rate",
        "0.2019%",
    )

with fraud_col2:
    st.metric(
        "Lowest Fraud Rate",
        "0.0702%",
    )

with fraud_col3:
    st.metric(
        "Period 5 Fraud Rate",
        "0.0878%",
    )

st.caption(
    "Fraud prevalence varies across the future periods. "
    "The observed fraud rate remains well below 1%, consistent "
    "with the highly imbalanced nature of the dataset."
)


# =========================================================
# FEATURE DRIFT MONITORING
# =========================================================

st.divider()

st.subheader("Feature Drift Monitoring")

st.write(
    """
    Feature drift is assessed by comparing feature distributions
    between the training period and the future test period using
    the Kolmogorov–Smirnov (KS) statistic. Larger values indicate
    greater distributional differences.
    """
)


drift_data = pd.DataFrame(
    {
        "Feature": [
            "V3",
            "V1",
            "V28",
            "V25",
            "V15",
            "V5",
            "V11",
            "V23",
            "V4",
            "V22",
        ],
        "KS Statistic": [
            0.3471,
            0.2652,
            0.2230,
            0.2188,
            0.1662,
            0.1581,
            0.1429,
            0.1378,
            0.1335,
            0.1227,
        ],
        "Drift Band": [
            "High",
            "High",
            "High",
            "High",
            "Moderate",
            "Moderate",
            "Moderate",
            "Moderate",
            "Moderate",
            "Moderate",
        ],
    }
)


st.markdown("#### Largest Distribution Shifts")

drift_chart = drift_data.set_index(
    "Feature"
)[["KS Statistic"]]

st.bar_chart(
    drift_chart,
    horizontal=True,
    x_label="KS Statistic",
)


st.dataframe(
    drift_data,
    use_container_width=True,
    hide_index=True,
)


st.caption(
    "Monitoring bands used here are practical interpretation bands: "
    "KS ≥ 0.20 indicates high drift, 0.10–0.20 indicates moderate drift, "
    "and lower values indicate smaller distributional changes. These "
    "bands are not universal statistical thresholds."
)


# =========================================================
# MONITORING INTERPRETATION
# =========================================================

st.markdown("#### Monitoring Interpretation")


st.info(
    """
    **Overall monitoring assessment**

    Model performance is relatively strong in the earlier future periods
    but declines in the later periods, particularly for F1 and PR-AUC.
    Several features also show noticeable distributional differences between
    the training period and future data.

    These findings are consistent with potential data or concept drift and
    suggest that model performance should be monitored over time rather
    than assumed to remain constant.

    However, drift does not establish causality. In addition, the final
    two temporal periods contain only 8 and 10 fraud cases respectively,
    making their performance estimates more sensitive to individual
    transactions. The later-period decline should therefore be treated
    as a monitoring signal rather than definitive proof of model failure.
    """
)


# =========================================================
# TRANSACTION SCORING
# =========================================================

st.divider()

st.subheader("Transaction Scoring")

st.write(
    "Enter the 30 transaction features below to obtain a prediction "
    "from the trained XGBoost model."
)


if not model_loaded:

    st.error(
        "The XGBoost model could not be loaded. "
        "Check that models/xgboost_fraud_model.joblib exists."
    )

    with st.expander("Technical error"):
        st.code(model_error)

else:

    with st.form("transaction_scoring_form"):

        st.markdown("### Transaction Details")

        col1, col2 = st.columns(2)

        with col1:

            time_value = st.number_input(
                "Time",
                value=0.0,
                step=1.0,
                help="Seconds elapsed from the first transaction.",
            )

        with col2:

            amount_value = st.number_input(
                "Amount",
                min_value=0.0,
                value=100.0,
                step=1.0,
                help="Transaction amount.",
            )

        st.markdown("### Anonymized Transaction Features")

        st.caption(
            "V1–V28 are anonymized/PCA-transformed features from the "
            "original dataset. For a meaningful prediction, enter values "
            "from the same feature space used during model training."
        )

        v_col1, v_col2, v_col3, v_col4 = st.columns(4)

        with v_col1:

            v1 = st.number_input(
                "V1",
                value=0.0,
                format="%.6f",
            )

            v2 = st.number_input(
                "V2",
                value=0.0,
                format="%.6f",
            )

            v3 = st.number_input(
                "V3",
                value=0.0,
                format="%.6f",
            )

            v4 = st.number_input(
                "V4",
                value=0.0,
                format="%.6f",
            )

            v5 = st.number_input(
                "V5",
                value=0.0,
                format="%.6f",
            )

            v6 = st.number_input(
                "V6",
                value=0.0,
                format="%.6f",
            )

            v7 = st.number_input(
                "V7",
                value=0.0,
                format="%.6f",
            )

        with v_col2:

            v8 = st.number_input(
                "V8",
                value=0.0,
                format="%.6f",
            )

            v9 = st.number_input(
                "V9",
                value=0.0,
                format="%.6f",
            )

            v10 = st.number_input(
                "V10",
                value=0.0,
                format="%.6f",
            )

            v11 = st.number_input(
                "V11",
                value=0.0,
                format="%.6f",
            )

            v12 = st.number_input(
                "V12",
                value=0.0,
                format="%.6f",
            )

            v13 = st.number_input(
                "V13",
                value=0.0,
                format="%.6f",
            )

            v14 = st.number_input(
                "V14",
                value=0.0,
                format="%.6f",
            )

        with v_col3:

            v15 = st.number_input(
                "V15",
                value=0.0,
                format="%.6f",
            )

            v16 = st.number_input(
                "V16",
                value=0.0,
                format="%.6f",
            )

            v17 = st.number_input(
                "V17",
                value=0.0,
                format="%.6f",
            )

            v18 = st.number_input(
                "V18",
                value=0.0,
                format="%.6f",
            )

            v19 = st.number_input(
                "V19",
                value=0.0,
                format="%.6f",
            )

            v20 = st.number_input(
                "V20",
                value=0.0,
                format="%.6f",
            )

            v21 = st.number_input(
                "V21",
                value=0.0,
                format="%.6f",
            )

        with v_col4:

            v22 = st.number_input(
                "V22",
                value=0.0,
                format="%.6f",
            )

            v23 = st.number_input(
                "V23",
                value=0.0,
                format="%.6f",
            )

            v24 = st.number_input(
                "V24",
                value=0.0,
                format="%.6f",
            )

            v25 = st.number_input(
                "V25",
                value=0.0,
                format="%.6f",
            )

            v26 = st.number_input(
                "V26",
                value=0.0,
                format="%.6f",
            )

            v27 = st.number_input(
                "V27",
                value=0.0,
                format="%.6f",
            )

            v28 = st.number_input(
                "V28",
                value=0.0,
                format="%.6f",
            )

        submitted = st.form_submit_button(
            "Score Transaction",
            use_container_width=True,
        )


    if submitted:

        transaction = pd.DataFrame(
            [[
                time_value,
                v1,
                v2,
                v3,
                v4,
                v5,
                v6,
                v7,
                v8,
                v9,
                v10,
                v11,
                v12,
                v13,
                v14,
                v15,
                v16,
                v17,
                v18,
                v19,
                v20,
                v21,
                v22,
                v23,
                v24,
                v25,
                v26,
                v27,
                v28,
                amount_value,
            ]],
            columns=FEATURES,
        )

        try:

            probability = float(
                model.predict_proba(transaction)[0, 1]
            )

            prediction = int(
                probability >= THRESHOLD
            )

            st.divider()

            st.subheader("Prediction Result")

            if probability >= THRESHOLD:

                risk_level = "High Risk"

                risk_message = (
                    "The predicted fraud probability is above the "
                    "operating threshold."
                )

            elif probability >= 0.20:

                risk_level = "Elevated Risk"

                risk_message = (
                    "The predicted probability is below the operating "
                    "threshold but indicates elevated risk."
                )

            else:

                risk_level = "Low Risk"

                risk_message = (
                    "The predicted fraud probability is below the "
                    "operating threshold."
                )


            risk_col1, risk_col2 = st.columns([2, 1])

            with risk_col1:

                st.markdown("#### Fraud Risk Assessment")

                if prediction == 1:

                    st.error(
                        f"**{risk_level}**"
                    )

                elif probability >= 0.20:

                    st.warning(
                        f"**{risk_level}**"
                    )

                else:

                    st.success(
                        f"**{risk_level}**"
                    )

                st.progress(
                    probability,
                    text=f"Fraud probability: {probability:.2%}",
                )

                st.caption(
                    f"Operating threshold: {THRESHOLD:.2f}"
                )


            with risk_col2:

                st.markdown("#### Risk Interpretation")

                st.metric(
                    "Probability",
                    f"{probability:.2%}",
                )

                st.metric(
                    "Threshold",
                    f"{THRESHOLD:.2f}",
                )


            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Fraud Probability",
                    f"{probability:.2%}",
                )

            with result_col2:

                st.metric(
                    "Decision Threshold",
                    f"{THRESHOLD:.2f}",
                )

            with result_col3:

                if prediction == 1:

                    st.metric(
                        "Decision",
                        "Potential Fraud",
                    )

                else:

                    st.metric(
                        "Decision",
                        "Likely Legitimate",
                    )


            if prediction == 1:

                st.error(
                    "Potential fraud detected. "
                    "This transaction exceeds the model's operating "
                    "threshold and should be reviewed."
                )

            else:

                st.success(
                    "Likely legitimate. "
                    "The predicted fraud probability is below the "
                    "operating threshold."
                )


            st.caption(
                risk_message
            )

            st.caption(
                "This prediction is generated by the trained XGBoost "
                "model using the 30 features supplied above."
            )


        except Exception as e:

            st.error(
                "The transaction could not be scored."
            )

            with st.expander("Technical error"):

                st.code(
                    str(e)
                )


# =========================================================
# BATCH TRANSACTION SCORING
# =========================================================

st.divider()

st.subheader("Batch Transaction Scoring")

st.write(
    "Upload a CSV containing the model's 30 input features "
    "to score multiple transactions at once."
)


uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
)


if uploaded_file is not None:

    try:

        uploaded_df = pd.read_csv(
            uploaded_file
        )

        if uploaded_df.empty:

            st.error(
                "The uploaded CSV is empty. "
                "Please upload a file containing transaction rows."
            )

        else:

            missing_features = [
                feature
                for feature in FEATURES
                if feature not in uploaded_df.columns
            ]

            if missing_features:

                st.error(
                    "The uploaded file is missing required model features."
                )

                st.write(
                    "Missing features:",
                    missing_features,
                )

            else:

                scoring_data = uploaded_df[
                    FEATURES
                ].copy()

                invalid_columns = []

                for feature in FEATURES:

                    converted = pd.to_numeric(
                        scoring_data[feature],
                        errors="coerce",
                    )

                    if converted.isna().any():

                        invalid_columns.append(
                            feature
                        )

                    scoring_data[feature] = converted


                if invalid_columns:

                    st.error(
                        "The uploaded file contains invalid or "
                        "non-numeric values."
                    )

                    st.write(
                        "Columns with invalid values:",
                        invalid_columns,
                    )

                else:

                    probabilities = model.predict_proba(
                        scoring_data
                    )[:, 1]

                    predictions = (
                        probabilities >= THRESHOLD
                    ).astype(int)

                    results = uploaded_df.copy()

                    results["Fraud_Probability"] = probabilities

                    results["Prediction"] = predictions

                    results["Decision"] = results[
                        "Prediction"
                    ].map(
                        {
                            0: "Likely Legitimate",
                            1: "Potential Fraud",
                        }
                    )


                    st.success(
                        f"Successfully scored "
                        f"{len(results):,} transactions."
                    )


                    batch_col1, batch_col2, batch_col3 = st.columns(3)

                    with batch_col1:

                        st.metric(
                            "Transactions Scored",
                            f"{len(results):,}",
                        )

                    with batch_col2:

                        st.metric(
                            "Potential Fraud",
                            f"{predictions.sum():,}",
                        )

                    with batch_col3:

                        fraud_rate = predictions.mean()

                        st.metric(
                            "Potential Fraud Rate",
                            f"{fraud_rate:.2%}",
                        )


                    st.markdown("#### Scored Transactions")

                    st.dataframe(
                        results.head(100),
                        use_container_width=True,
                        hide_index=True,
                    )


                    if len(results) > 100:

                        st.caption(
                            "Showing the first 100 rows. "
                            "Download the complete results below."
                        )


                    download_data = results.to_csv(
                        index=False
                    ).encode("utf-8")


                    st.download_button(
                        label="Download Scored Results",
                        data=download_data,
                        file_name="fraud_scored_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


    except pd.errors.EmptyDataError:

        st.error(
            "The uploaded file contains no readable CSV data."
        )

    except Exception as e:

        st.error(
            "The uploaded file could not be processed."
        )

        with st.expander("Technical error"):

            st.code(
                str(e)
            )