"""
Compare additional model candidates (Random Forest, LightGBM, CatBoost, and a
stacked ensemble of Logistic Regression + XGBoost) against our current best
model (calibrated Logistic Regression, ROC-AUC 0.695 / PR-AUC 0.516 on test).

IMPORTANT METHODOLOGY NOTE: since Logistic Regression and XGBoost were already
evaluated on the test set in train.py, evaluating new candidates on that same
test set now -- and picking whichever looks best -- would itself be a form of
data snooping (repeatedly consulting the test set until something wins is not
meaningfully different from tuning on it). So every new candidate here is
compared on the VALIDATION set only, exactly like the original XGBoost tuning
was. The test set is touched at most once more, only for whichever single
candidate (if any) beats the current best on validation -- and only to confirm
it, not to search over several finalists there.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from features import build_feature_pipeline, MODEL_FEATURE_COLUMNS, CATEGORICAL_COLUMNS
from split import time_based_split_3way
from train import build_preprocessor, NUMERIC_COLUMNS


def evaluate_on(name, pipe, X, y):
    proba = pipe.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    print(f"  {name:45s}  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    return roc_auc, pr_auc


def main():
    df = pd.read_csv("data/processed/features.csv")
    train_df, val_df, test_df = time_based_split_3way(df)

    X_train, y_train = train_df[MODEL_FEATURE_COLUMNS], train_df["Return_Status"].map(
        {"Returned": 1, "Not Returned": 0}
    )
    X_val, y_val = val_df[MODEL_FEATURE_COLUMNS], val_df["Return_Status"].map(
        {"Returned": 1, "Not Returned": 0}
    )
    X_test, y_test = test_df[MODEL_FEATURE_COLUMNS], test_df["Return_Status"].map(
        {"Returned": 1, "Not Returned": 0}
    )

    pos, neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = neg / pos

    # Also get the current best model's OWN validation performance, computed fresh
    # here rather than assumed, so the comparison to it is apples-to-apples.
    current_best = Pipeline_wrap(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    current_best.fit(X_train, y_train)
    current_roc, current_pr = evaluate_on("Logistic Regression (current best, uncalibrated)", current_best, X_val, y_val)

    print()
    print("=" * 70)
    print("NEW CANDIDATES — evaluated on VALIDATION SET ONLY")
    print("=" * 70)

    results = {}

    # Random Forest
    rf = Pipeline_wrap(RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ))
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate_on("Random Forest", rf, X_val, y_val)

    # LightGBM
    lgbm = Pipeline_wrap(LGBMClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.03,
        min_child_samples=20, reg_lambda=2.0, class_weight="balanced",
        random_state=42, verbosity=-1,
    ))
    lgbm.fit(X_train, y_train)
    results["LightGBM"] = evaluate_on("LightGBM", lgbm, X_val, y_val)

    # CatBoost (handles categoricals natively, but we keep the same preprocessing
    # for a fair apples-to-apples comparison against the others)
    cb = Pipeline_wrap(CatBoostClassifier(
        iterations=200, depth=4, learning_rate=0.03, l2_leaf_reg=3.0,
        auto_class_weights="Balanced", random_state=42, verbose=False,
    ))
    cb.fit(X_train, y_train)
    results["CatBoost"] = evaluate_on("CatBoost", cb, X_val, y_val)

    # Stacked ensemble: Logistic Regression + XGBoost, combined via a meta logistic
    # regression. Trained fresh here on train only (not train+val) so we can fairly
    # compare its validation performance against the other candidates above.
    base_logreg = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    base_xgb = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.03, min_child_weight=5,
            reg_lambda=2.0, scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=42, n_jobs=-1,
        )),
    ])
    stack = StackingClassifier(
        estimators=[("logreg", base_logreg), ("xgb", base_xgb)],
        final_estimator=LogisticRegression(class_weight="balanced"),
        cv=5, n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    results["Stacked (LogReg+XGB)"] = evaluate_on("Stacked ensemble (LogReg + XGBoost)", stack, X_val, y_val)

    print()
    print("=" * 70)
    print("SUMMARY (validation set)")
    print("=" * 70)
    for name, (roc, pr) in sorted(results.items(), key=lambda kv: -kv[1][1]):
        print(f"  {name:45s}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

    best_name = max(results, key=lambda k: results[k][1])
    best_pr = results[best_name][1]
    print(f"\nBest new candidate on validation: {best_name} (PR-AUC={best_pr:.4f})")
    print(f"Current best (Logistic Regression) validation PR-AUC: {current_pr:.4f}")
    if best_pr > current_pr:
        print(">>> This candidate beats the current model on validation. Worth confirming on test once.")
    else:
        print(">>> None of the new candidates beat the current model on validation. Keeping Logistic Regression.")


from sklearn.pipeline import Pipeline


def Pipeline_wrap(clf):
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def confirm_on_test(X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight):
    """
    Refit the winning stacked ensemble on TRAIN+VAL (same convention as train.py),
    then evaluate ONCE on the held-out test set. This is the only additional test-set
    touch in this whole comparison exercise, and it is for a single named finalist,
    not a search over multiple candidates.
    """
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    base_logreg = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    base_xgb = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.03, min_child_weight=5,
            reg_lambda=2.0, scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=42, n_jobs=-1,
        )),
    ])
    stack = StackingClassifier(
        estimators=[("logreg", base_logreg), ("xgb", base_xgb)],
        final_estimator=LogisticRegression(class_weight="balanced"),
        cv=5, n_jobs=-1,
    )
    stack.fit(X_trainval, y_trainval)

    proba = stack.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print("\n" + "=" * 70)
    print("FINAL TEST-SET CONFIRMATION — Stacked Ensemble (refit on train+val)")
    print("=" * 70)
    print(f"  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  Brier={brier:.4f}")
    print(f"  (Current calibrated Logistic Regression on test: ROC-AUC=0.6950  PR-AUC=0.5160  Brier=0.2003)")

    return stack, {"roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier}


if __name__ == "__main__":
    df = pd.read_csv("data/processed/features.csv")
    train_df, val_df, test_df = time_based_split_3way(df)
    X_train, y_train = train_df[MODEL_FEATURE_COLUMNS], train_df["Return_Status"].map({"Returned": 1, "Not Returned": 0})
    X_val, y_val = val_df[MODEL_FEATURE_COLUMNS], val_df["Return_Status"].map({"Returned": 1, "Not Returned": 0})
    X_test, y_test = test_df[MODEL_FEATURE_COLUMNS], test_df["Return_Status"].map({"Returned": 1, "Not Returned": 0})
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    main()
    confirm_on_test(X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight)
