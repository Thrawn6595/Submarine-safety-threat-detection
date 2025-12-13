from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DEFAULT_SEED, SONAR_TARGET_COL

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_sonar_raw() -> pd.DataFrame:
    path = DATA_DIR / "raw" / "sonar.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw sonar.csv at {path}")
    # sonar.csv is typically headerless: 60 features + 1 label col
    return pd.read_csv(path, header=None)

def split_train_val_test(
    df: pd.DataFrame,
    target_col: int = SONAR_TARGET_COL,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = DEFAULT_SEED,
    stratify: bool = True,
):
    y = df.iloc[:, target_col]
    X = df.drop(df.columns[target_col], axis=1)

    strat = y if stratify else None

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=seed, stratify=strat
    )

    rel_test = test_size / (test_size + val_size)
    strat_tmp = y_tmp if stratify else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, random_state=seed, stratify=strat_tmp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def assign_feature_and_outcome_names(
    df: pd.DataFrame,
    outcome_col_index: int,
    outcome_name: str = "outcome",
    feature_prefix: str = "feature",
):
    """
    Assigns deterministic column names to a headerless dataframe.

    Features: feature_1, feature_2, ...
    Outcome: renamed explicitly (default: 'outcome')
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
):
    """
    Maps string outcome values to integers using an explicit mapping.
    Example: {'R': 0, 'M': 1}
    """
    df = df.copy()

    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found")

    unmapped = set(df[outcome_col].unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(f"Unmapped outcome values found: {unmapped}")

    df[outcome_col] = df[outcome_col].map(mapping).astype(int)
    return df

