from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from submarine_threat_detection.data import load_sonar_raw, split_train_test
from submarine_threat_detection.preprocessing import prepare_data_for_algorithm
from submarine_threat_detection.train_utils import train_model_with_cv, create_cv_results_table
from submarine_threat_detection.config import DEFAULT_SEED

from sklearn.linear_model import LogisticRegression

def main():
    df = load_sonar_raw()

    # Sonar: last column is label
    X_train, X_test, y_train, y_test = split_train_test(
    df,
    target_col="outcome",
    test_size=0.2,
    seed=DEFAULT_SEED,
    stratify=True,
)


    prepared = prepare_data_for_algorithm(X_train, X_val, X_test, y_train, y_val, y_test)

    # Baseline model + small grid (expand later)
    model = LogisticRegression(max_iter=2000, random_state=DEFAULT_SEED)
    grid = {"C": [0.1, 1.0, 10.0]}

    res = train_model_with_cv(
        model=model,
        param_grid=grid,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_val=prepared.X_val,
        y_val=prepared.y_val,
        scoring="f1",
        cv_folds=5,
        seed=DEFAULT_SEED,
    )

    table = create_cv_results_table([res])
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
