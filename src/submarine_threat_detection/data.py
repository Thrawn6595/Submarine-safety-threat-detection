import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_SEED


def load_sonar_raw(path: str = "data/raw/sonar.csv") -> pd.DataFrame:
    """
    Load the raw sonar dataset (no headers in source file).
    """
    return pd.read_csv(path, header=None)


def assign_feature_and_outcome_names(
    df: pd.DataFrame,
    outcome_col_index: int,
    outcome_name: str = "outcome",
    feature_prefix: str = "feature",
) -> pd.DataFrame:
    """
    Assign deterministic column names to a headerless dataframe.
    """
    df = df.copy()
    n_cols = df.shape[1]

    cols = []
    for i in range(n_cols):
        if i == outcome_col_index:
            cols.append(outcome_name)
        else:
            cols.append(f"{feature_prefix}_{i + 1}")

    df.columns = cols
    return df


def map_outcome_to_int(
    df: pd.DataFrame,
    outcome_col: str,
    mapping: dict,
) -> pd.DataFrame:
    """
    Map string outcome labels to integers using an explicit mapping.
    """
    df = df.copy()

    unmapped = set(df[outcome_col].unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(f"Unmapped outcome values found: {unmapped}")

    df[outcome_col] = df[outcome_col].map(mapping).astype(int)
    return df


def split_train_test(
    df: pd.DataFrame,
    target_col: str = "outcome",
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
    stratify: bool = True,
):
    """
    Train/Test split for small datasets.
    Cross-validation happens entirely inside TRAIN.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])

    strat = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )
