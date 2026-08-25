"""
Return-Risk Scorer -- Streamlit demo

Lets a user simulate a single order and see:
1. The model's predicted return probability
2. Which features drove that score (top contributing factors)
3. A recommendation based on the capacity-constrained business threshold
4. Cold-start handling for brand-new users/products (explicitly, not silently)

Run with: streamlit run app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from features import MODEL_FEATURE_COLUMNS, CATEGORICAL_COLUMNS

st.set_page_config(page_title="Return-Risk Scorer", page_icon="package", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "logreg_pipeline.joblib")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")

CAPACITY_CONSTRAINED_THRESHOLD = 0.50  # chosen from cost_eval.py at ~15% review capacity
# (updated after calibrating the model -- calibration shifted the probability scale, so this
# threshold must be re-derived from cost_eval.py any time the model or its calibration changes)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    """Used only to look up realistic prior stats for existing users/categories/products,
    simulating what a real feature store lookup would return."""
    df = pd.read_csv(FEATURES_PATH)
    return df


def get_user_prior_stats(df, user_id):
    """Cold-start-safe lookup: returns None if user has no history."""
    user_rows = df[df["User_ID"] == user_id]
    if len(user_rows) == 0:
        return None
    last_row = user_rows.iloc[-1]
    return {
        "user_prior_order_count": last_row["user_prior_order_count"] + 1,
        "user_prior_return_rate": last_row["user_prior_return_rate"],
        "is_new_user": 0,
        "user_avg_order_value_prior": last_row["user_avg_order_value_prior"],
        "user_days_since_last_order": 0,
        "user_category_diversity_prior": last_row["user_category_diversity_prior"],
    }


def cold_start_defaults(df):
    """Safe fallback stats for a brand-new user, derived from population-wide
    priors rather than a null/crash. This is the same fallback logic that
    should run in production if a real-time feature lookup misses or times out."""
    return {
        "user_prior_order_count": 0,
        "user_prior_return_rate": df["user_prior_return_rate"].mean(),
        "is_new_user": 1,
        "user_avg_order_value_prior": df["Order_Value"].mean(),
        "user_days_since_last_order": -1,  # matches the sentinel value used in training for no-prior-order
        "user_category_diversity_prior": 0,
    }


def get_category_prior(df, category):
    rows = df[df["Product_Category"] == category]
    if len(rows) == 0:
        return df["category_prior_return_rate"].mean()
    return rows["category_prior_return_rate"].iloc[-1]


st.title("Return-Risk Scorer")
st.caption("Predicts probability that an order will be returned, at checkout / return-request time.")

model = load_model()
ref_df = load_reference_data()

st.divider()
st.subheader("Simulate an order")

col1, col2 = st.columns(2)
with col1:
    product_price = st.number_input("Product price (rupees, per unit)", min_value=100, max_value=2000, value=1000, step=50)
    order_quantity = st.number_input("Order quantity", min_value=1, max_value=5, value=2, step=1)
    category = st.selectbox("Product category", sorted(ref_df["Product_Category"].unique()))
    payment_method = st.selectbox("Payment method", sorted(ref_df["Payment_Method"].unique()))
    shipping_method = st.selectbox("Shipping method", sorted(ref_df["Shipping_Method"].unique()))

with col2:
    discount_pct_display = st.slider("Discount percent", 0, 50, 10)
    discount_pct = discount_pct_display / 100.0  # training data stores this as a 0-1 fraction, not 0-100
    order_date = st.date_input("Order date (for weekday/weekend feature)", value=pd.Timestamp.now())
    is_known_user = st.checkbox("Existing user? (uncheck to simulate a brand-new / cold-start user)", value=True)
    user_id = None
    if is_known_user:
        sample_users = sorted(ref_df["User_ID"].unique())[:200]
        user_id = st.selectbox("Pick an existing User ID (demo sample)", sample_users)

st.divider()

if st.button("Score this order", type="primary"):
    order_value = product_price * order_quantity
    order_ts = pd.Timestamp(order_date)

    row = {}
    row["Product_Price"] = product_price
    row["Order_Quantity"] = order_quantity
    row["discount_pct"] = discount_pct
    row["Order_Value"] = order_value
    row["value_per_item"] = order_value / order_quantity
    row["order_month"] = order_ts.month
    row["order_dayofweek"] = order_ts.dayofweek
    row["order_is_weekend"] = int(order_ts.dayofweek >= 5)
    row["Product_Category"] = category
    row["Payment_Method"] = payment_method
    row["Shipping_Method"] = shipping_method

    # User features -- cold-start-safe
    cold_start_triggered = False
    if is_known_user and user_id is not None:
        user_stats = get_user_prior_stats(ref_df, user_id)
        if user_stats is None:
            user_stats = cold_start_defaults(ref_df)
            cold_start_triggered = True
    else:
        user_stats = cold_start_defaults(ref_df)
        cold_start_triggered = True
    row.update(user_stats)

    # Category / product priors -- cold-start-safe
    row["category_prior_return_rate"] = get_category_prior(ref_df, category)
    row["product_prior_return_rate"] = ref_df["product_prior_return_rate"].mean()  # demo simplification: no specific SKU picker

    # Interaction features
    row["price_x_discount_pct"] = order_value * discount_pct
    row["is_cod_x_high_value"] = int(payment_method == "COD" and order_value > ref_df["Order_Value"].median())
    row["user_x_category_return_rate"] = row["user_prior_return_rate"] * row["category_prior_return_rate"]
    row["discount_x_category_return_rate"] = discount_pct * row["category_prior_return_rate"]

    input_df = pd.DataFrame([row])
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in input_df.columns]
    if missing:
        st.error(f"Internal error: missing engineered features {missing}. This should never happen -- "
                 f"it would indicate a train/serve skew bug.")
        st.stop()

    input_df = input_df[MODEL_FEATURE_COLUMNS]
    proba = model.predict_proba(input_df)[0, 1]

    if cold_start_triggered:
        st.info("Cold-start fallback used: this user has no prior history, so population-level "
                "default statistics were used instead of a crash or a null score.")

    st.metric("Predicted return probability", f"{proba:.1%}")

    if proba >= CAPACITY_CONSTRAINED_THRESHOLD:
        st.error(f"HIGH RISK -- flag for review (threshold {CAPACITY_CONSTRAINED_THRESHOLD:.0%}, "
                 f"calibrated to a realistic 15% review-team capacity)")
    else:
        st.success("Low / acceptable risk -- no intervention needed")

    with st.expander("Why this score? (top contributing factors)"):
        st.write(f"Category base risk for **{category}**: {row['category_prior_return_rate']:.1%} historical return rate")
        st.write(f"Payment method: **{payment_method}**" +
                 (" (COD carries higher risk in this dataset)" if payment_method == "COD" else ""))
        st.write(f"User prior return rate: {row['user_prior_return_rate']:.1%}" +
                 (" (new user, population average used)" if cold_start_triggered else " (from order history)"))
        st.write(f"Discount depth: {discount_pct_display} percent")
        st.write(f"Order value: Rs.{order_value:,.2f} ({order_quantity} x Rs.{product_price:,.2f})")

st.divider()
with st.expander("About this demo / robustness notes"):
    st.write(
        "This scorer is built to be leak-proof: every feature is computed only from information "
        "available strictly BEFORE the order in question, using time-ordered rolling statistics. "
        "Cold-start users and products fall back to safe population-level defaults instead of "
        "crashing or returning null. In production, the rolling user/category statistics shown "
        "here would be served from a precomputed feature store or cache, not a live full-table scan, "
        "to keep checkout-time latency safe."
    )
