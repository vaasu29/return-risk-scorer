"""
Time-based train/test split.

We deliberately do NOT use a random split. A random split would let orders
from, say, March train the model while orders from January test it — that's
not how deployment works, and it lets the model implicitly see "the future"
via user/category/product aggregates that were computed with the full dataset's
time ordering in mind. A time-based split (train on the past, test on the
most recent slice) mirrors real production deployment and is the more
defensible choice for a fraud/risk system.
"""

import pandas as pd


def time_based_split(df: pd.DataFrame, date_col: str = "Order_Date", test_frac: float = 0.2):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def time_based_split_3way(df: pd.DataFrame, date_col: str = "Order_Date",
                           val_frac: float = 0.15, test_frac: float = 0.2):
    """
    Three-way time-ordered split: train / validation / test.

    - train: used to fit model parameters
    - validation: used ONLY for hyperparameter selection (e.g. XGBoost grid search)
    - test: touched exactly once, only to report final numbers. Never used for
      any decision-making (model choice, threshold choice, hyperparameters).

    This avoids the subtle leakage of "tuning against the test set," which
    quietly inflates reported metrics even when the train/test split itself
    is correct.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(test_start * (1 - val_frac))

    train = df.iloc[:val_start].copy()
    val = df.iloc[val_start:test_start].copy()
    test = df.iloc[test_start:].copy()
    return train, val, test


if __name__ == "__main__":
    df = pd.read_csv("data/processed/features.csv", parse_dates=["Order_Date"])
    train, test = time_based_split(df)
    print(f"Train: {len(train)} rows, {train['Order_Date'].min()} to {train['Order_Date'].max()}")
    print(f"Test:  {len(test)} rows, {test['Order_Date'].min()} to {test['Order_Date'].max()}")
    print(f"Train target rate: {train['target'].mean():.3f}")
    print(f"Test target rate:  {test['target'].mean():.3f}")

    print("\n--- 3-way split ---")
    train3, val3, test3 = time_based_split_3way(df)
    print(f"Train: {len(train3)} rows, {train3['Order_Date'].min()} to {train3['Order_Date'].max()}")
    print(f"Val:   {len(val3)} rows, {val3['Order_Date'].min()} to {val3['Order_Date'].max()}")
    print(f"Test:  {len(test3)} rows, {test3['Order_Date'].min()} to {test3['Order_Date'].max()}")
    print(f"Train target rate: {train3['target'].mean():.3f} | Val: {val3['target'].mean():.3f} | Test: {test3['target'].mean():.3f}")
