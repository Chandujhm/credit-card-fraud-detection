import pandas as pd

FILE_PATH = "data/raw/creditcard.csv"


def main():
    df = pd.read_csv(FILE_PATH)

    fraud = df[df["Class"] == 1]
    legitimate = df[df["Class"] == 0]

    print("=" * 60)
    print("FRAUD TRANSACTION ANALYSIS")
    print("=" * 60)

    print("\nTransaction counts:")
    print(f"Total transactions: {len(df):,}")
    print(f"Legitimate transactions: {len(legitimate):,}")
    print(f"Fraudulent transactions: {len(fraud):,}")

    print("\nFraud transaction amount statistics:")
    print(fraud["Amount"].describe())

    print("\nLegitimate transaction amount statistics:")
    print(legitimate["Amount"].describe())

    print("\nAverage transaction amount:")
    print(f"Legitimate: {legitimate['Amount'].mean():.2f}")
    print(f"Fraud:      {fraud['Amount'].mean():.2f}")

    print("\nMedian transaction amount:")
    print(f"Legitimate: {legitimate['Amount'].median():.2f}")
    print(f"Fraud:      {fraud['Amount'].median():.2f}")

    print("\nTotal transaction amount:")
    print(f"Legitimate: {legitimate['Amount'].sum():,.2f}")
    print(f"Fraud:      {fraud['Amount'].sum():,.2f}")

    print("\nMaximum transaction amount:")
    print(f"Legitimate: {legitimate['Amount'].max():.2f}")
    print(f"Fraud:      {fraud['Amount'].max():.2f}")


if __name__ == "__main__":
    main()