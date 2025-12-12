from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

def build_preprocessor() -> Pipeline:
    # Sonar: numeric features only
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: Pipeline
    label_encoder: Optional[LabelEncoder]

class ModelPreprocessor:
    """
    Backwards-compatible wrapper expected by older notebooks.
    Keeps leakage-safe behaviour: fit on train only.
    """
    def __init__(self, encode_labels: bool = True):
        self.encode_labels = encode_labels
        self.preprocessor = build_preprocessor()
        self.label_encoder: Optional[LabelEncoder] = None
        self._is_fit = False

    def fit(self, X_train: pd.DataFrame, y_train=None):
        self.preprocessor.fit(X_train)
        self._is_fit = True

        if self.encode_labels and y_train is not None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y_train)
        return self

    def transform(self, X: pd.DataFrame):
        if not self._is_fit:
            raise RuntimeError("ModelPreprocessor must be fit on training data before transform().")
        return self.preprocessor.transform(X)

    def transform_y(self, y):
        if not self.encode_labels:
            return np.asarray(y)
        if self.label_encoder is None:
            raise RuntimeError("Label encoder not fit. Call fit(..., y_train=...) first.")
        return self.label_encoder.transform(y)

def prepare_data_for_algorithm(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_val,
    y_test,
    encode_labels: bool = True,
) -> PreparedData:
    mp = ModelPreprocessor(encode_labels=encode_labels)
    mp.fit(X_train, y_train=y_train)

    X_train_t = mp.transform(X_train)
    X_val_t = mp.transform(X_val)
    X_test_t = mp.transform(X_test)

    y_train_t = mp.transform_y(y_train) if encode_labels else np.asarray(y_train)
    y_val_t = mp.transform_y(y_val) if encode_labels else np.asarray(y_val)
    y_test_t = mp.transform_y(y_test) if encode_labels else np.asarray(y_test)

    return PreparedData(
        X_train=X_train_t, X_val=X_val_t, X_test=X_test_t,
        y_train=np.asarray(y_train_t), y_val=np.asarray(y_val_t), y_test=np.asarray(y_test_t),
        preprocessor=mp.preprocessor,
        label_encoder=mp.label_encoder
    )
