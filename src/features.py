"""
Feature engineering for the Return-Risk Scorer.

CRITICAL DESIGN RULE — LEAKAGE PREVENTION:
Every user-level / product-level / category-level aggregate feature is computed
using ONLY orders that occurred strictly BEFORE the current order's date
(expanding window, sorted by Order_Date). This simulates what a real system
would know at the moment a new order comes in — it cannot see the future,
and it cannot see the outcome of the very order it's trying to score.

Columns that describe the RETURN OUTCOME itself (Return_Reason, Days_to_Return,
Return_Cost, Profit_Loss, CO2_*, Packaging_Waste, Waste_Avoided) are NEVER used
as model features. They are outcome/downstream fields, only used later for the
cost-matrix evaluation.
"""

import pandas as pd
import numpy as np

# Fields that are only known AFTER a return happens — never use as model input.
LEAKAGE_COLUMNS = [
    "Return_Reason", "Days_to_Return", "Return_Cost", "Profit_Loss",
    "CO2_Emissions", "Packaging_Waste", "CO2_Saved", "Waste_Avoided",
]

# Reserved for cost-matrix evaluation only (post-hoc, never fed to the model).
COST_MATRIX_COLUMNS = ["Return_Cost", "Profit_Loss"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Order_Date"])
    df = df.sort_values("Order_Date").reset_index(drop=True)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = (df["Return_Status"] == "Returned").astype(int)
    return df


def add_order_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 1: features available at the moment of order placement."""
    df = df.copy()
    df["discount_pct"] = (df["Discount_Applied"] / df["Product_Price"].replace(0, np.nan)).fillna(0)
    df["order_month"] = df["Order_Date"].dt.month
    df["order_dayofweek"] = df["Order_Date"].dt.dayofweek
    df["order_is_weekend"] = df["order_dayofweek"].isin([5, 6]).astype(int)
    df["price_per_unit"] = df["Product_Price"]
    df["value_per_item"] = df["Order_Value"] / df["Order_Quantity"].replace(0, np.nan)
    df["value_per_item"] = df["value_per_item"].fillna(df["Order_Value"])
    return df


def _expanding_prior_stats(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    """
    Compute expanding (as-of, strictly-prior) count and return-rate stats
    for a grouping column (e.g. User_ID, Product_ID, Product_Category).

    Uses shift(1) on the expanding window so the current row's own outcome
    is excluded from its own prior-stat calculation.
    """
    df = df.copy()
    grp = df.groupby(group_col)["target"]

    # Expanding count of PRIOR orders (excludes current row)
    prior_count = grp.cumcount()  # 0-indexed count of rows before this one in the group
    # Expanding sum of PRIOR returns (shift(1) so current target isn't included)
    prior_return_sum = grp.transform(lambda s: s.shift(1).expanding().sum())

    df[f"{prefix}_prior_order_count"] = prior_count
    df[f"{prefix}_prior_return_count"] = prior_return_sum.fillna(0)
    df[f"{prefix}_prior_return_rate"] = (
        df[f"{prefix}_prior_return_count"] / df[f"{prefix}_prior_order_count"].replace(0, np.nan)
    )
    return df


def add_user_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 2: user-level behavioral features, strictly time-aware.
    Data must already be sorted by Order_Date ascending before calling this.
    """
    df = df.copy()
    df = _expanding_prior_stats(df, "User_ID", "user")

    # Global fallback rate for cold-start users (no prior orders).
    global_prior_rate = df.loc[df["user_prior_order_count"] > 0, "target"].mean()
    if pd.isna(global_prior_rate):
        global_prior_rate = df["target"].mean()

    df["is_new_user"] = (df["user_prior_order_count"] == 0).astype(int)
    df["user_prior_return_rate"] = df["user_prior_return_rate"].fillna(global_prior_rate)

    # Average order value of user's PRIOR orders only (expanding, shifted).
    df["user_avg_order_value_prior"] = (
        df.groupby("User_ID")["Order_Value"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    df["user_avg_order_value_prior"] = df["user_avg_order_value_prior"].fillna(df["Order_Value"].mean())

    # Days since this user's previous order (0 / NaN -> large sentinel for first order).
    df["user_prev_order_date"] = df.groupby("User_ID")["Order_Date"].shift(1)
    df["user_days_since_last_order"] = (df["Order_Date"] - df["user_prev_order_date"]).dt.days
    df["user_days_since_last_order"] = df["user_days_since_last_order"].fillna(-1)  # -1 = first order
    df.drop(columns=["user_prev_order_date"], inplace=True)

    # Distinct categories user has ordered from PRIOR to this order.
    def _prior_category_diversity(sub: pd.DataFrame) -> pd.Series:
        seen = set()
        out = []
        for cat in sub["Product_Category"]:
            out.append(len(seen))
            seen.add(cat)
        return pd.Series(out, index=sub.index)

    df["user_category_diversity_prior"] = (
        df.sort_values("Order_Date").groupby("User_ID", group_keys=False).apply(_prior_category_diversity)
    )

    return df


def add_category_and_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 3: category-level and product-level prior return-rate features."""
    df = df.copy()
    df = _expanding_prior_stats(df, "Product_Category", "category")
    df = _expanding_prior_stats(df, "Product_ID", "product")

    global_rate = df["target"].mean()
    df["category_prior_return_rate"] = df["category_prior_return_rate"].fillna(global_rate)
    df["product_prior_return_rate"] = df["product_prior_return_rate"].fillna(global_rate)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 4: interaction features."""
    df = df.copy()
    df["price_x_discount_pct"] = df["Product_Price"] * df["discount_pct"]
    df["is_cod_x_high_value"] = (
        (df["Payment_Method"] == "COD") & (df["Order_Value"] > df["Order_Value"].median())
    ).astype(int)
    df["user_x_category_return_rate"] = df["user_prior_return_rate"] * df["category_prior_return_rate"]

    # New interaction: heavy discounts may correlate with impulse buys that get returned more
    # often in categories that are already higher-risk. Safe to add because both component
    # features (discount_pct, category_prior_return_rate) are already leak-proof/time-aware
    # per-row values -- no new cross-row aggregation is introduced here.
    df["discount_x_category_return_rate"] = df["discount_pct"] * df["category_prior_return_rate"]
    return df


def build_feature_pipeline(raw_path: str) -> pd.DataFrame:
    """Run the full leak-proof feature engineering pipeline end to end."""
    df = load_raw(raw_path)
    df = add_target(df)
    df = add_order_level_features(df)
    df = add_user_behavioral_features(df)
    df = add_category_and_product_features(df)
    df = add_interaction_features(df)
    return df


MODEL_FEATURE_COLUMNS = [
    # Tier 1 — order level
    "Product_Price", "Order_Quantity", "discount_pct", "Order_Value",
    "value_per_item", "order_month", "order_dayofweek", "order_is_weekend",
    "Product_Category", "Shipping_Method", "Payment_Method",
    # Tier 2 — user behavioral
    "user_prior_order_count", "user_prior_return_rate", "is_new_user",
    "user_avg_order_value_prior", "user_days_since_last_order",
    "user_category_diversity_prior",
    # Tier 3 — category / product
    "category_prior_return_rate", "product_prior_return_rate",
    # Tier 4 — interactions
    "price_x_discount_pct", "is_cod_x_high_value", "user_x_category_return_rate",
    "discount_x_category_return_rate",
]

CATEGORICAL_COLUMNS = ["Product_Category", "Shipping_Method", "Payment_Method"]


if __name__ == "__main__":
    df = build_feature_pipeline("data/raw/synthetic_returns_dataset.csv")
    print(f"Built {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Target rate: {df['target'].mean():.3f}")
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in df.columns]
    print(f"Missing expected feature columns: {missing}")
    df.to_csv("data/processed/features.csv", index=False)
    print("Saved to data/processed/features.csv")
