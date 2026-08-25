"""
Train baseline (Logistic Regression) and upgraded (XGBoost) models for
return-risk scoring.

Always trained/evaluated on the TIME-BASED split (see src/split.py) —
never a random split, to avoid simulating information the model wouldn't
have in production.

Class imbalance handled via class_weight='balanced' (logistic regression)
and scale_pos_weight (XGBoost) — NOT via SMOTE, since SMOTE synthesizes
rows by interpolating between existing points, which is a poor fit for
data that includes rolling time-aware aggregates (interpolating would
blur the temporal meaning of user_prior_return_rate etc.). Class weighting
avoids this problem entirely while still correcting for the ~29%/71% imbalance.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, brier_score_loss,
)
from xgboost import XGBClassifier

from features import build_feature_pipeline, MODEL_FEATURE_COLUMNS, CATEGORICAL_COLUMNS
from split import time_based_split_3way

NUMERIC_COLUMNS = [c for c in MODEL_FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def train_logistic(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train, **xgb_kwargs) -> Pipeline:
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos

    defaults = dict(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(xgb_kwargs)

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(**defaults)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def tune_xgboost_quick(X_train, y_train, X_val, y_val):
    """Small manual grid search over the params most likely to fix overfitting
    on a ~4000-row dataset (shallower trees, stronger regularization, fewer rounds)."""
    grid = [
        dict(n_estimators=100, max_depth=2, learning_rate=0.05, min_child_weight=10, reg_lambda=3.0),
        dict(n_estimators=150, max_depth=3, learning_rate=0.03, min_child_weight=5, reg_lambda=2.0),
        dict(n_estimators=200, max_depth=3, learning_rate=0.02, min_child_weight=8, reg_lambda=3.0),
        dict(n_estimators=100, max_depth=4, learning_rate=0.03, min_child_weight=10, reg_lambda=5.0, subsample=0.7),
        dict(n_estimators=250, max_depth=2, learning_rate=0.03, min_child_weight=5, reg_lambda=1.5),
        # widened grid: a couple more shapes to check before locking in
        dict(n_estimators=300, max_depth=2, learning_rate=0.02, min_child_weight=8, reg_lambda=2.5),
        dict(n_estimators=150, max_depth=3, learning_rate=0.04, min_child_weight=6, reg_lambda=1.0, colsample_bytree=0.6),
        dict(n_estimators=200, max_depth=4, learning_rate=0.02, min_child_weight=12, reg_lambda=4.0, subsample=0.75),
    ]
    best_auc, best_params, best_model = -1, None, None
    for params in grid:
        model = train_xgboost(X_train, y_train, **params)
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        print(f"  params={params} -> ROC-AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, best_params, best_model = auc, params, model
    print(f"Best: {best_params} (ROC-AUC={best_auc:.4f})")
    return best_model


def calibrate_model(fitted_pipeline_ctor, X_trainval, y_trainval):
    """
    Wrap the already-tuned estimator in CalibratedClassifierCV (5-fold internal
    cross-fitting, sigmoid/Platt scaling). This does NOT touch the test set and
    does NOT calibrate a model on the same rows it was fit on -- CalibratedClassifierCV
    internally splits train/val folds itself, refitting the base estimator on each fold.

    WHY THIS MATTERS FOR "HONEST METRICS": our cost-matrix evaluation (cost_eval.py)
    treats predict_proba() output as a real probability of return, and multiplies it
    against actual rupee costs. If the raw scores are poorly calibrated (e.g. systematically
    over- or under-confident), the cost estimates built on top of them are quietly wrong even
    if ROC-AUC/PR-AUC look fine, since those two metrics only care about RANKING, not the
    actual probability values. Calibration doesn't change ranking (AUC is usually near-identical)
    but makes the probability values themselves trustworthy for cost-weighted decisions.
    """
    calibrated = CalibratedClassifierCV(fitted_pipeline_ctor, method="sigmoid", cv=5)
    calibrated.fit(X_trainval, y_trainval)
    return calibrated


def evaluate_model(name: str, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC (average precision): {pr_auc:.4f}   "
          f"Brier score (lower=better calibrated): {brier:.4f}")

    # Precision/recall at a few candidate thresholds
    thresholds_to_check = [0.3, 0.4, 0.5, 0.6, 0.7]
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, proba)
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Flagged %':>10}")
    for t in thresholds_to_check:
        preds = (proba >= t).astype(int)
        tp = ((preds == 1) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        fn = ((preds == 0) & (y_test == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        flagged_pct = preds.mean() * 100
        print(f"{t:>10.2f} {prec:>10.3f} {rec:>10.3f} {flagged_pct:>9.1f}%")

    print(f"\nFull classification report at threshold=0.5:")
    print(classification_report(y_test, (proba >= 0.5).astype(int), target_names=["Not Returned", "Returned"]))

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier, "proba": proba}


def main():
    print("Building feature pipeline...")
    df = build_feature_pipeline("data/raw/synthetic_returns_dataset.csv")

    print("Time-based train/validation/test split...")
    train_df, val_df, test_df = time_based_split_3way(df, val_frac=0.15, test_frac=0.2)
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")
    print(f"Train target rate: {train_df['target'].mean():.3f} | "
          f"Val target rate: {val_df['target'].mean():.3f} | "
          f"Test target rate: {test_df['target'].mean():.3f}")

    X_train, y_train = train_df[MODEL_FEATURE_COLUMNS], train_df["target"]
    X_val, y_val = val_df[MODEL_FEATURE_COLUMNS], val_df["target"]
    X_test, y_test = test_df[MODEL_FEATURE_COLUMNS], test_df["target"]

    print("\nTuning XGBoost on VALIDATION set only (test set untouched until final report)...")
    xgb_tuning_model = tune_xgboost_quick(X_train, y_train, X_val, y_val)
    best_params = xgb_tuning_model.named_steps["clf"].get_xgb_params()
    # keep only the params we actually swept, to avoid passing back internal xgboost state
    swept_keys = ["n_estimators", "max_depth", "learning_rate", "min_child_weight",
                  "reg_lambda", "reg_alpha", "subsample", "colsample_bytree"]
    best_params = {k: v for k, v in best_params.items() if k in swept_keys and v is not None}

    print(f"\nRefitting XGBoost on TRAIN+VAL combined with selected hyperparameters: {best_params}")
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    xgb = train_xgboost(X_trainval, y_trainval, **best_params)
    xgb_results = evaluate_model("XGBoost (tuned on val, refit on train+val) — TEST SET", xgb, X_test, y_test)

    print("\nRefitting Logistic Regression on TRAIN+VAL for the final model too (fair comparison, same data)...")
    logreg = train_logistic(X_trainval, y_trainval)
    logreg_results = evaluate_model("Logistic Regression (refit on train+val) — TEST SET", logreg, X_test, y_test)

    print("\nCalibrating Logistic Regression (5-fold internal cross-fit, sigmoid scaling)...")
    logreg_uncalibrated_ctor = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    logreg_calibrated = calibrate_model(logreg_uncalibrated_ctor, X_trainval, y_trainval)
    logreg_cal_results = evaluate_model(
        "Logistic Regression CALIBRATED (refit on train+val) — TEST SET", logreg_calibrated, X_test, y_test
    )
    print(f"Brier score change from calibration: {logreg_results['brier']:.4f} -> {logreg_cal_results['brier']:.4f} "
          f"({'improved' if logreg_cal_results['brier'] < logreg_results['brier'] else 'no improvement'})")

    # Use the calibrated model as the one shipped to the app/demo, since cost_eval.py
    # depends on predict_proba() being a trustworthy probability, not just a good ranking score.
    joblib.dump(logreg_calibrated, "models/logreg_pipeline.joblib")
    joblib.dump(xgb, "models/xgb_pipeline.joblib")
    test_df = test_df.copy()
    test_df["proba_logreg"] = logreg_cal_results["proba"]
    test_df["proba_logreg_uncalibrated"] = logreg_results["proba"]
    test_df["proba_xgb"] = xgb_results["proba"]
    test_df.to_csv("data/processed/test_with_scores.csv", index=False)

    print("\nSaved CALIBRATED logreg + xgb models to models/, scored test set to data/processed/test_with_scores.csv")

    # Feature importance for XGBoost (quick look)
    feature_names = (
        NUMERIC_COLUMNS +
        list(xgb.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLUMNS))
    )
    importances = xgb.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    print("\nTop 10 XGBoost feature importances:")
    print(imp_df.head(10).to_string(index=False))
    imp_df.to_csv("data/processed/feature_importance.csv", index=False)


if __name__ == "__main__":
    main()
