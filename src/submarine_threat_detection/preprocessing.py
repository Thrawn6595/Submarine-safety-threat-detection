from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

def build_preprocessor() -> Pipeline:
    # Sonar is numeric features; baseline: scale.
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

def prepare_data_for_algorithm(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_val,
    y_test,
    encode_labels: bool = True,
) -> PreparedData:
    pre = build_preprocessor()

    # FIT ONLY ON TRAIN
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    le = None
    if encode_labels:
        le = LabelEncoder()
        y_train_t = le.fit_transform(y_train)
        y_val_t = le.transform(y_val)
        y_test_t = le.transform(y_test)
    else:
        y_train_t, y_val_t, y_test_t = y_train, y_val, y_test

    return PreparedData(
        X_train=X_train_t, X_val=X_val_t, X_test=X_test_t,
        y_train=np.asarray(y_train_t), y_val=np.asarray(y_val_t), y_test=np.asarray(y_test_t),
        preprocessor=pre, label_encoder=le
    )
