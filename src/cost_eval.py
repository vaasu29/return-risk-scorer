"""
Cost-matrix evaluation for the Return-Risk Scorer.

This is the "honest metrics including false-positive cost" requirement
from the track brief. Accuracy/F1 alone hide the fact that a False Positive
(flagging a genuine order as high-risk) and a False Negative (missing an
actual return) have very DIFFERENT dollar costs to the merchant. We use the
dataset's own Return_Cost and Profit_Loss columns to build a real, data-driven
cost matrix instead of guessing arbitrary numbers.

COST DEFINITIONS (derived from data, not invented):
- True Positive (correctly flagged a return): merchant can intervene
  (e.g. extra verification, restocking-fee prompt, or size-guidance nudge)
  BEFORE the return happens. We model this as recovering a fraction of the
  return cost -- an intervention won't stop every flagged return, but a
  realistic assumption is it prevents some share of them. We call this the
  INTERVENTION_RECOVERY_RATE and treat it as a documented assumption to be
  tuned per merchant risk appetite, not a hard fact.
- False Positive (flagged a genuine, would-not-return order as risky):
  the cost is friction -- e.g. an extra verification step or a delayed
  COD confirmation. We model this as a small fixed FRICTION_COST per
  flagged order (lost conversion / customer annoyance), NOT the full
  return cost, since the order was never actually going to be returned.
- False Negative (missed an actual return): the merchant absorbs the
  full historical Return_Cost for that order, exactly as they do today
  with no model at all.
- True Negative (correctly left a genuine order alone): zero cost.

This lets us report NET EXPECTED SAVINGS per threshold: how much cheaper
is deploying the model (at a given threshold) than doing nothing at all.
"""

import pandas as pd
import numpy as np

# ---- Documented, tunable business assumptions (NOT invented per-row facts) ----
INTERVENTION_RECOVERY_RATE = 0.35   # fraction of flagged true-risk returns that get prevented
FRICTION_COST_PER_FALSE_POSITIVE = 25.0  # rupees-equivalent cost of unnecessary friction on a genuine order


def compute_cost_at_threshold(df: pd.DataFrame, proba_col: str, threshold: float,
                               intervention_recovery_rate: float = INTERVENTION_RECOVERY_RATE,
                               friction_cost: float = FRICTION_COST_PER_FALSE_POSITIVE) -> dict:
    """
    df must contain: target (0/1), Return_Cost, and the given proba_col.
    Returns a dict of counts and costs at this threshold.
    """
    preds = (df[proba_col] >= threshold).astype(int)
    y = df["target"]

    tp_mask = (preds == 1) & (y == 1)
    fp_mask = (preds == 1) & (y == 0)
    fn_mask = (preds == 0) & (y == 1)
    tn_mask = (preds == 0) & (y == 0)

    tp, fp, fn, tn = tp_mask.sum(), fp_mask.sum(), fn_mask.sum(), tn_mask.sum()

    # Cost if we deploy the model at this threshold
    tp_cost = df.loc[tp_mask, "Return_Cost"].sum() * (1 - intervention_recovery_rate)
    fp_cost = fp * friction_cost
    fn_cost = df.loc[fn_mask, "Return_Cost"].sum()
    tn_cost = 0.0
    model_total_cost = tp_cost + fp_cost + fn_cost + tn_cost

    # Baseline cost: cost of doing NOTHING (no model at all) -- merchant absorbs
    # every actual return's full cost, with zero friction cost anywhere.
    baseline_cost = df.loc[y == 1, "Return_Cost"].sum()

    savings = baseline_cost - model_total_cost
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": precision, "recall": recall,
        "flagged_pct": preds.mean() * 100,
        "model_total_cost": model_total_cost,
        "baseline_cost_do_nothing": baseline_cost,
        "net_savings": savings,
        "savings_pct_of_baseline": (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0,
    }


def sweep_thresholds(df: pd.DataFrame, proba_col: str, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.85, 0.05)
    rows = [compute_cost_at_threshold(df, proba_col, t) for t in thresholds]
    return pd.DataFrame(rows)


def find_optimal_threshold(df: pd.DataFrame, proba_col: str) -> dict:
    sweep = sweep_thresholds(df, proba_col, thresholds=np.arange(0.05, 0.95, 0.01))
    best_row = sweep.loc[sweep["net_savings"].idxmax()]
    return best_row.to_dict(), sweep


def find_optimal_threshold_with_capacity(df: pd.DataFrame, proba_col: str, max_flagged_pct: float) -> dict:
    """
    Same as find_optimal_threshold, but only considers thresholds where the
    percentage of orders flagged for review is <= max_flagged_pct.

    WHY THIS MATTERS: a merchant's fraud/returns review team has a fixed
    headcount. A threshold that is "cheapest on paper" but flags 68% of all
    orders is not deployable -- no ops team can manually review two-thirds
    of traffic. Optimizing cost alone, with no capacity constraint, produces
    a mathematically correct but operationally useless answer. This function
    finds the best NET SAVINGS achievable within a realistic review capacity.
    """
    sweep = sweep_thresholds(df, proba_col, thresholds=np.arange(0.05, 0.95, 0.01))
    feasible = sweep[sweep["flagged_pct"] <= max_flagged_pct]
    if feasible.empty:
        raise ValueError(
            f"No threshold keeps flagged_pct <= {max_flagged_pct}%. "
            f"Lowest achievable flagged_pct is {sweep['flagged_pct'].min():.1f}%."
        )
    best_row = feasible.loc[feasible["net_savings"].idxmax()]
    return best_row.to_dict(), sweep


if __name__ == "__main__":
    df = pd.read_csv("data/processed/test_with_scores.csv")

    print("="*70)
    print("COST-MATRIX ASSUMPTIONS (documented, tunable)")
    print("="*70)
    print(f"Intervention recovery rate on flagged true returns: {INTERVENTION_RECOVERY_RATE:.0%}")
    print(f"Friction cost per false positive: Rs.{FRICTION_COST_PER_FALSE_POSITIVE:.2f}")
    print()

    REVIEW_CAPACITY_PCT = 15.0  # documented assumption: ops team can review ~15% of order volume

    for model_name, proba_col in [("Logistic Regression", "proba_logreg"), ("XGBoost", "proba_xgb")]:
        print("="*70)
        print(f"{model_name}")
        print("="*70)
        sweep = sweep_thresholds(df, proba_col)
        display_cols = ["threshold", "precision", "recall", "flagged_pct",
                         "model_total_cost", "net_savings", "savings_pct_of_baseline"]
        print(sweep[display_cols].round(2).to_string(index=False))

        best, _ = find_optimal_threshold(df, proba_col)
        print(f"\nUNCONSTRAINED optimal threshold for {model_name}: {best['threshold']:.2f}")
        print(f"  Precision: {best['precision']:.3f} | Recall: {best['recall']:.3f}")
        print(f"  Flags {best['flagged_pct']:.1f}% of orders for review  <-- often NOT operationally feasible")
        print(f"  Net savings: Rs.{best['net_savings']:,.2f} ({best['savings_pct_of_baseline']:.1f}% reduction)")

        try:
            best_capped, _ = find_optimal_threshold_with_capacity(df, proba_col, REVIEW_CAPACITY_PCT)
            print(f"\nCAPACITY-CONSTRAINED optimal (max {REVIEW_CAPACITY_PCT:.0f}% of orders flagged):")
            print(f"  Threshold: {best_capped['threshold']:.2f}")
            print(f"  Precision: {best_capped['precision']:.3f} | Recall: {best_capped['recall']:.3f}")
            print(f"  Flags {best_capped['flagged_pct']:.1f}% of orders for review  <-- deployable")
            print(f"  Baseline cost (do nothing): Rs.{best_capped['baseline_cost_do_nothing']:,.2f}")
            print(f"  Model cost at this threshold: Rs.{best_capped['model_total_cost']:,.2f}")
            print(f"  Net savings: Rs.{best_capped['net_savings']:,.2f} ({best_capped['savings_pct_of_baseline']:.1f}% reduction)")
        except ValueError as e:
            print(f"\nCAPACITY-CONSTRAINED optimal: {e}")
        print()

    # Save full sweep for both models to CSV for the README/report
    all_sweeps = []
    for model_name, proba_col in [("logreg", "proba_logreg"), ("xgb", "proba_xgb")]:
        s = sweep_thresholds(df, proba_col)
        s["model"] = model_name
        all_sweeps.append(s)
    pd.concat(all_sweeps).to_csv("data/processed/cost_matrix_sweep.csv", index=False)
    print("Saved full threshold sweep to data/processed/cost_matrix_sweep.csv")

