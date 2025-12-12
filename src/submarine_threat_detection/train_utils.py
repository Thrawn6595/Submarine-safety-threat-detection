from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import DEFAULT_SEED

@dataclass
class CVRunResult:
    model_name: str
    best_params: Dict[str, Any]
    cv_best_score: float
    val_metrics: Dict[str, float]
    fitted_model: BaseEstimator

def train_model_with_cv(
    model: BaseEstimator,
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scoring: str = "f1",
    cv_folds: int = 5,
    seed: int = DEFAULT_SEED,
) -> CVRunResult:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    preds = best.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1": float(f1_score(y_val, preds)),
    }

    # Only compute AUC if model supports predict_proba
    if hasattr(best, "predict_proba"):
        proba = best.predict_proba(X_val)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_val, proba))

    return CVRunResult(
        model_name=best.__class__.__name__,
        best_params=gs.best_params_,
        cv_best_score=float(gs.best_score_),
        val_metrics=metrics,
        fitted_model=best,
    )

def create_cv_results_table(results: List[CVRunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "model": r.model_name,
            "cv_best_score": r.cv_best_score,
            **{f"val_{k}": v for k, v in r.val_metrics.items()},
            "best_params": r.best_params,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    # Rank by val_f1 if present, else val_accuracy
    sort_col = "val_f1" if "val_f1" in df.columns else "val_accuracy"
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)
