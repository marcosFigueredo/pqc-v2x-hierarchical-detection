import numpy as np
from sklearn.datasets import make_classification

from src.model_training import get_models, train_and_eval_models
from src.run_pipeline import REAL_RUN_MODEL_NAMES


def test_get_models_returns_four_named_models():
    models = get_models()
    assert set(models.keys()) == {"lightgbm", "logreg", "random_forest", "mlp"}


def test_train_and_eval_models_fits_and_predicts():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_classes=3,
        n_clusters_per_class=1, random_state=42,
    )
    X_train, X_val = X[:200], X[200:]
    y_train, y_val = y[:200], y[200:]

    results = train_and_eval_models(X_train, y_train, X_val, y_val)

    assert set(results.keys()) == {"lightgbm", "logreg", "random_forest", "mlp"}
    for name, res in results.items():
        assert res["eval_predictions"].shape[0] == X_val.shape[0]
        assert set(np.unique(res["eval_predictions"])).issubset({0, 1, 2})


def test_full_run_roster_is_explicit_and_covers_all_baselines():
    assert REAL_RUN_MODEL_NAMES == ["lightgbm", "logreg", "random_forest", "mlp"]
