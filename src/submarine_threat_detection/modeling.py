from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from .config import DEFAULT_SEED
from .train_utils import train_model_with_cv, create_results_table, evaluate_binary_classifier, CVRunResult

# -----------------------
# Model configuration
# -----------------------

def get_model_configurations(seed: int = DEFAULT_SEED):
    """
    Returns a list of (name, estimator, param_grid, scoring).
    Expand this over time.
    """
    return [
        ("LogisticRegression",
         LogisticRegression(max_iter=2000, random_state=seed),
         {"C": [0.1, 1.0, 10.0]},
         "f1"),
        ("SVC",
         SVC(probability=True, random_state=seed),
         {"C": [0.5, 1.0, 2.0], "kernel": ["rbf", "linear"]},
         "f1"),
        ("RandomForestClassifier",
         RandomForestClassifier(random_state=seed),
         {"n_estimators": [200, 500], "max_depth": [None, 5, 10]},
         "f1"),
    ]

# -----------------------
# Training + evaluation wrappers (old notebook API)
# -----------------------

def tune_and_evaluate_model(
    model: BaseEstimator,
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scoring: str = "f1",
    cv_folds: int = 5,
    seed: int = DEFAULT_SEED,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> CVRunResult:
    return train_model_with_cv(
        model=model,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        scoring=scoring,
        cv_folds=cv_folds,
        seed=seed,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )

def evaluate_final_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> Dict[str, float]:
    return evaluate_binary_classifier(model, X_test, y_test, fp_cost=fp_cost, fn_cost=fn_cost)

def create_cv_results_table(results: List[CVRunResult], rank_by: str = "val_f1") -> pd.DataFrame:
    return create_results_table(results, rank_by=rank_by)

def create_test_results_table(test_results: List[Dict[str, Any]], rank_by: str = "test_f1") -> pd.DataFrame:
    """
    Accepts list of dicts: {"model": name, "test_metrics": {...}, "best_params": {...}}
    """
    rows = []
    for r in test_results:
        row = {"model": r["model"], **{f"test_{k}": v for k, v in r["test_metrics"].items()}, "best_params": r.get("best_params", {})}
        rows.append(row)
    df = pd.DataFrame(rows)
    if rank_by not in df.columns:
        rank_by = "test_f1" if "test_f1" in df.columns else "test_accuracy"
    return df.sort_values(rank_by, ascending=False).reset_index(drop=True)

# -----------------------
# Plotting (optional): keep stubs for now
# -----------------------

def plot_cv_comparison(*args, **kwargs):
    raise NotImplementedError("Plotting not wired yet. Use the results table for now.")

def plot_test_comparison(*args, **kwargs):
    raise NotImplementedError("Plotting not wired yet. Use the results table for now.")
