from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import DEFAULT_SEED


# -----------------------
# Scorers (selection metric)
# -----------------------

F2_SCORER = make_scorer(fbeta_score, beta=2, zero_division=0)  # recall-weighted


# -----------------------
# Result container
# -----------------------

@dataclass
class CVRunResult:
    model_name: str
    best_params: Dict[str, Any]
    cv_best_score: float
    # Kept name "val_metrics" for backward compatibility with your notebook API,
    # but it now stores CV aggregate metrics (mean/std), not holdout-val metrics.
    val_metrics: Dict[str, float]
    fitted_model: BaseEstimator


# -----------------------
# Helpers
# -----------------------

def _safe_predict_scores(model: BaseEstimator, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Return a continuous score for positive class if available.
    Preference: predict_proba[:, 1] else decision_function.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores is not None:
            return scores
    return None

def _compute_cost_from_confusion(cm: np.ndarray, fp_cost: float, fn_cost: float) -> float:
    # binary confusion matrix: [[tn, fp], [fn, tp]]
    if cm.size != 4:
        return float("nan")
    tn, fp, fn, tp = cm.ravel()
    return float(fp_cost * fp + fn_cost * fn)

def evaluate_binary_classifier(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> Dict[str, float]:
    """
    Compute a standard bundle of metrics for binary classification.
    Returns threshold metrics + probability metrics when available.
    """
    preds = model.predict(X)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "f2": float(fbeta_score(y, preds, beta=2, zero_division=0)),
    }

    cm = confusion_matrix(y, preds)
    metrics["cost"] = _compute_cost_from_confusion(cm, fp_cost=fp_cost, fn_cost=fn_cost)

    scores = _safe_predict_scores(model, X)
    if scores is not None:
        # These accept either probabilities or decision scores
        metrics["roc_auc"] = float(roc_auc_score(y, scores))
        metrics["pr_auc"] = float(average_precision_score(y, scores))

    return metrics


# -----------------------
# Main: train + CV (train_test_cv strategy)
# -----------------------

def train_model_with_cv(
    model: BaseEstimator,
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    # Kept args for backward compatibility; ignored in train_test_cv.
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    scoring: Any = F2_SCORER,
    cv_folds: int = 5,
    seed: int = DEFAULT_SEED,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> CVRunResult:
    """
    Train + tune on TRAIN only using CV.
    Returns:
      - best estimator refit on full TRAIN
      - best params
      - best CV selection score
      - aggregate CV metrics (mean/std) for all metrics we compute

    Note:
      X_val/y_val are accepted for old call sites but ignored by design.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,   # selection metric (default: F2)
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_

    # Fold-by-fold evaluation for full metric bundle
    fold_metrics: List[Dict[str, float]] = []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        m = clone(best)
        m.fit(X_tr, y_tr)
        fold_metrics.append(evaluate_binary_classifier(m, X_va, y_va, fp_cost=fp_cost, fn_cost=fn_cost))

    mdf = pd.DataFrame(fold_metrics)

    # Aggregate mean/std
    agg: Dict[str, float] = {}
    for col in mdf.columns:
        agg[f"cv_{col}_mean"] = float(mdf[col].mean())
        agg[f"cv_{col}_std"] = float(mdf[col].std(ddof=1))

    # Back-compat field name: store CV aggregate metrics in val_metrics
    return CVRunResult(
        model_name=best.__class__.__name__,
        best_params=gs.best_params_,
        cv_best_score=float(gs.best_score_),
        val_metrics=agg,
        fitted_model=best,
    )


def create_cv_results_table(results: List[CVRunResult], rank_by: str = "cv_f2_mean") -> pd.DataFrame:
    """
    Convert CVRunResult objects into a ranked results table.
    rank_by examples:
      - "cv_recall_mean"
      - "cv_precision_mean"
      - "cv_f2_mean" (default)
      - "cv_pr_auc_mean"
      - "cv_cost_mean" (lower is better; handle separately if needed)
    """
    rows = []
    for r in results:
        row = {
            "model": r.model_name,
            "cv_best_score": r.cv_best_score,
            **r.val_metrics,  # contains cv_*_mean and cv_*_std
            "best_params": r.best_params,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if rank_by not in df.columns:
        # fall back safely
        rank_by = "cv_f2_mean" if "cv_f2_mean" in df.columns else df.columns[0]

    # If ranking by cost, lower is better
    ascending = True if "cost" in rank_by else False

    return df.sort_values(rank_by, ascending=ascending).reset_index(drop=True)
