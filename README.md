# Advanced Credit Card Fraud Detection

An end-to-end machine learning system for detecting potentially fraudulent credit card transactions under severe class imbalance, temporal drift, and asymmetric error costs.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small proportion of all transactions.

The objective of this project is to build a fraud detection system that:

- Detects fraudulent transactions effectively
- Minimizes unnecessary false-positive alerts
- Uses a realistic chronological validation strategy
- Prevents temporal leakage
- Evaluates performance on future unseen transactions
- Monitors model performance and feature drift over time
- Provides an interactive Streamlit dashboard for individual and batch scoring

## Dataset

The project uses the **Credit Card Fraud Detection** dataset containing European card transactions.

Dataset characteristics:

- 284,807 transactions
- 30 model input features
- 492 fraudulent transactions
- Fraud rate: approximately 0.173%
- Features `V1`–`V28` are anonymized/PCA-transformed
- `Time` represents elapsed seconds from the first transaction
- `Amount` represents transaction amount
- `Class` is the target variable

The raw dataset is intentionally excluded from version control.

## Project Structure

```text
Credit Card Fraud Detection/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── xgboost_fraud_model.joblib
│   └── model_metadata.json
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── reports/
│
├── src/
│   ├── split_data.py
│   ├── xgboost_model.py
│   ├── evaluate_saved_model.py
│   ├── train_xgboost_dev.py
│   └── threshold_validation.py
│
├── tests/
│
├── .gitignore
├── requirements.txt
└── README.md