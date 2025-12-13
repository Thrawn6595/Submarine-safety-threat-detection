from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from submarine_threat_detection.config import DEFAULT_SEED
from submarine_threat_detection.data import load_sonar_raw, split_train_test
from submarine_threat_detection.preprocessing import prepare_data_for_algorithm
from submarine_threat_detection.train_utils import (
    train_model_with_cv,
    create_cv_results_table,
    evaluate_binary_classifier,
)

from sklearn.linear_model import LogisticRegression


def main():
    # 1) Load + standardise schema (feature_* + outcome 0/1)
    df = load_sonar_raw()

    # 2) Hold-out test; CV happens inside train
    X_train, X_test, y_train, y_test = split_train_test(
        df,
        target_col="outcome",
        test_size=0.2,
        seed=DEFAULT_SEED,
        stratify=True,
    )

    # 3) Preprocess (fit on train only)
    prepared = prepare_data_for_algorithm(
        X_train, X_test, y_train, y_test,
        encode_labels=False,  # outcome already int from loader
    )

    # 4) Baseline model + grid (expand later)
    model = LogisticRegression(max_iter=2000, random_state=DEFAULT_SEED)
    grid = {"C": [0.1, 1.0, 10.0]}

    # 5) Train with CV on train only (no separate val)
    res = train_model_with_cv(
        model=model,
        param_grid=grid,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        scoring="f1",     # switch to F2 later for recall priority
        cv_folds=5,
        seed=DEFAULT_SEED,
        fp_cost=1.0,
        fn_cost=5.0,
    )

    # 6) Print ranked CV results table (single model for now)
    table = create_cv_results_table([res], rank_by="cv_best_score")
    print(table.to_string(index=False))

    # 7) Evaluate champion on held-out test
    test_metrics = evaluate_binary_classifier(
        res.fitted_model,
        prepared.X_test,
        prepared.y_test,
        fp_cost=1.0,
        fn_cost=5.0,
    )
    print("\nTEST metrics (champion):")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
