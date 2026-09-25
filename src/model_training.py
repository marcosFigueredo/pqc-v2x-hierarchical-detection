import numpy as np

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight


def get_models(seed=42):
    return {
        "lightgbm": LGBMClassifier(n_estimators=200, random_state=seed, n_jobs=-1, verbosity=-1),
        "logreg": LogisticRegression(max_iter=200, random_state=seed, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed),
    }


def train_and_eval_models(X_train, y_train, X_eval, y_eval, seed=42, model_names=None):
    sample_weight = compute_sample_weight("balanced", y_train)
    all_models = get_models(seed=seed)
    selected = all_models if model_names is None else {
        name: all_models[name] for name in model_names
    }
    results = {}
    for name, model in selected.items():
        if name == "mlp":
            # sklearn's MLPClassifier.fit has no sample_weight parameter,
            # so it trains unweighted; documented limitation, baseline only.
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        eval_predictions = model.predict(X_eval)
        results[name] = {
            "model": model,
            "eval_predictions": eval_predictions,
            "convergence": _convergence_status(model),
        }
    return results


def _convergence_status(model):
    """Report whether an iterative model hit its iteration cap.

    Reviewers of the flat-vs-hierarchical comparison can otherwise not tell
    whether a weak baseline is weak because the task is hard or because the
    optimizer stopped early. LightGBM and random forest are not iterative in
    this sense, so they report `None`.
    """
    max_iter = getattr(model, "max_iter", None)
    if max_iter is None:
        return None

    n_iter = getattr(model, "n_iter_", None)
    if n_iter is None:
        return {"max_iter": int(max_iter), "n_iter": None, "converged": None}

    # LogisticRegression exposes n_iter_ as one entry per class/ovr problem.
    n_iter_max = int(np.max(np.asarray(n_iter)))
    return {
        "max_iter": int(max_iter),
        "n_iter": n_iter_max,
        "converged": bool(n_iter_max < int(max_iter)),
    }
