from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ----------------------------
# Preprocessor builder
# ----------------------------

def build_preprocessor() -> Pipeline:
    """
    Sonar: numeric features only.
    Standardize features using training statistics only (fit on train).
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])


# ----------------------------
# Output container (train/test)
# ----------------------------

@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: Pipeline
    label_encoder: Optional[LabelEncoder]


# ----------------------------
# Preprocessor wrapper
# ----------------------------

class ModelPreprocessor:
    """
    Leakage-safe preprocessing wrapper.
    Fit on TRAIN only, transform any split with the fitted transform.

    encode_labels:
      - If True, fit a LabelEncoder on y_train and transform labels.
      - If False, labels are passed through as numpy arrays.
    """
    def __init__(self, encode_labels: bool = True):
        self.encode_labels = encode_labels
        self.preprocessor: Pipeline = build_preprocessor()
        self.label_encoder: Optional[LabelEncoder] = None
        self._is_fit: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: Any = None) -> "ModelPreprocessor":
        self.preprocessor.fit(X_train)
        self._is_fit = True

        if self.encode_labels and y_train is not None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y_train)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("ModelPreprocessor must be fit on training data before transform().")
        return self.preprocessor.transform(X)

    def transform_y(self, y: Any) -> np.ndarray:
        if not self.encode_labels:
            return np.asarray(y)
        if self.label_encoder is None:
            raise RuntimeError("Label encoder not fit. Call fit(..., y_train=...) first.")
        return self.label_encoder.transform(y)


# ----------------------------
# Main API (train/test)
# ----------------------------

def prepare_data_for_algorithm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Any,
    y_test: Any,
    *,
    encode_labels: bool = True,
) -> PreparedData:
    """
    Train/Test preparation for modeling (CV-ready).

    - Fit preprocessing on TRAIN only.
    - Transform TEST using fitted preprocessing.
    - Optionally encode labels (usually False if outcome already 0/1).

    Returns PreparedData containing numpy arrays and fitted artifacts.
    """
    mp = ModelPreprocessor(encode_labels=encode_labels)
    mp.fit(X_train, y_train=y_train)

    X_train_t = mp.transform(X_train)
    X_test_t = mp.transform(X_test)

    y_train_t = mp.transform_y(y_train) if encode_labels else np.asarray(y_train)
    y_test_t = mp.transform_y(y_test) if encode_labels else np.asarray(y_test)

    return PreparedData(
        X_train=np.asarray(X_train_t),
        X_test=np.asarray(X_test_t),
        y_train=np.asarray(y_train_t).astype(int),
        y_test=np.asarray(y_test_t).astype(int),
        preprocessor=mp.preprocessor,
        label_encoder=mp.label_encoder,
    )