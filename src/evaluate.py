import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def family_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {"accuracy": accuracy, "macro_f1": macro_f1, "confusion_matrix": cm, "labels": labels}


def subclass_metrics(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels))
    per_class = {
        label: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for label, p, r, f in zip(labels, precision, recall, f1)
    }
    return {"macro_f1": macro_f1, "per_class": per_class, "labels": labels}


def compare_real_synthetic(df, y_true_col, y_pred_col, is_real_trace_col="is_real_trace"):
    real_df = df[df[is_real_trace_col] == True]
    synthetic_df = df[df[is_real_trace_col] == False]
    return {
        "real": family_metrics(real_df[y_true_col], real_df[y_pred_col]),
        "synthetic": family_metrics(synthetic_df[y_true_col], synthetic_df[y_pred_col]),
    }


# Sentinel used when the cascade produces no subtype at all for a row,
# because Stage 1 never routed it to Stage 2. PQ-V2X subtype labels are the
# non-negative integers 0..23, so -1 can never collide with a real label and
# can never accidentally score as correct. It is kept numeric rather than a
# string so the prediction array stays a single sortable dtype — a mixed
# int/str object array makes scikit-learn's label handling raise.
NOT_ROUTED = -1


def cascade_predictions(routed_mask, routed_predictions):
    """Expand Stage 2's routed-only predictions to one prediction per row.

    Rows the cascade never routed get `NOT_ROUTED`, which is not a class
    label and therefore always scores as an error. Kept separate from
    `end_to_end_subtype_metrics` so the bootstrap can resample the same
    full-length prediction vector the metric is computed from.
    """
    routed_mask = np.asarray(routed_mask, dtype=bool)
    routed_predictions = np.asarray(routed_predictions)
    predictions = np.full(routed_mask.shape, NOT_ROUTED, dtype=routed_predictions.dtype)
    predictions[routed_mask] = routed_predictions
    return predictions


def end_to_end_subtype_metrics(true_subtypes, routed_mask, routed_predictions):
    """Score the cascade over every truly PQ/Hybrid row, not only routed ones.

    `subclass_metrics` on the routed subset answers "when Stage 2 is given a
    row, how well does it label it?" — it charges the cascade for routing
    false positives but is silent about routing false negatives, because a
    true PQ/Hybrid attack that Stage 1 sent elsewhere simply leaves the
    evaluation set. For a security detector that is the error that matters
    most, so this function keeps those rows in and gives them a prediction
    that is always wrong.

    Args:
        true_subtypes: subtype labels of ALL rows whose true family is
            PQ/Hybrid (i.e. the rows with r*_i = 1).
        routed_mask: boolean mask over those same rows, True where Stage 1
            actually routed the row to Stage 2 (r_i = 1).
        routed_predictions: Stage 2's predictions for the routed rows, in the
            order they appear under `routed_mask`.

    A flat classifier has no routing stage and therefore never abstains; to
    compare against it, score its predictions over the same `true_subtypes`
    with a `routed_mask` that is all True.
    """
    true_subtypes = np.asarray(true_subtypes)
    routed_mask = np.asarray(routed_mask, dtype=bool)
    routed_predictions = np.asarray(routed_predictions)

    if routed_mask.shape != true_subtypes.shape:
        raise ValueError(
            f"routed_mask {routed_mask.shape} must align with true_subtypes "
            f"{true_subtypes.shape}"
        )
    if routed_predictions.shape[0] != int(routed_mask.sum()):
        raise ValueError(
            f"routed_predictions has {routed_predictions.shape[0]} rows but "
            f"routed_mask selects {int(routed_mask.sum())}"
        )

    predictions = cascade_predictions(routed_mask, routed_predictions)

    # Labels come from ground truth only: NOT_ROUTED must never become a
    # scored class, it must only ever cost recall on the true class.
    labels = sorted(set(true_subtypes.tolist()))
    if NOT_ROUTED in labels:
        raise ValueError(
            f"sentinel {NOT_ROUTED!r} collides with a real subtype label; "
            "pick a sentinel outside the label space"
        )
    metrics = subclass_metrics(true_subtypes, predictions, labels=labels)
    metrics["n_total"] = int(true_subtypes.shape[0])
    metrics["n_routed"] = int(routed_mask.sum())
    metrics["n_never_routed"] = int((~routed_mask).sum())
    return metrics


def trivial_baselines(y_train, y_eval, seed=42):
    """Macro-F1 floor for a task: majority-class and stratified-random.

    Without these, a reader cannot tell whether a macro-F1 of 0.58 on a
    four-class problem whose majority class holds 60% of the rows is a result
    or an artifact of the metric.
    """
    y_train = np.asarray(y_train)
    y_eval = np.asarray(y_eval)
    labels = sorted(set(y_train.tolist()) | set(y_eval.tolist()))

    train_labels, train_counts = np.unique(y_train, return_counts=True)
    majority_label = train_labels[int(np.argmax(train_counts))]
    majority_pred = np.full(y_eval.shape, majority_label, dtype=y_eval.dtype)

    rng = np.random.default_rng(seed)
    stratified_pred = rng.choice(
        train_labels, size=y_eval.shape[0], p=train_counts / train_counts.sum()
    )

    return {
        "majority_class": {
            "label": majority_label.item() if isinstance(majority_label, np.generic) else majority_label,
            "accuracy": float((y_eval == majority_pred).mean()),
            "macro_f1": float(f1_score(y_eval, majority_pred, average="macro", labels=labels, zero_division=0)),
        },
        "stratified_random": {
            "accuracy": float((y_eval == stratified_pred).mean()),
            "macro_f1": float(f1_score(y_eval, stratified_pred, average="macro", labels=labels, zero_division=0)),
            "seed": int(seed),
        },
    }


def paired_bootstrap_macro_f1(y_true, pred_a, pred_b, labels=None, n_boot=1000, seed=42):
    """Bootstrap CI for macro-F1(a) - macro-F1(b) on the same rows.

    The headline claim of this study is a difference of roughly 0.02 macro-F1
    between two classifiers scored on identical rows. Resampling those rows
    with replacement and recomputing both scores on each resample gives that
    difference an interval instead of a bare point estimate. Because both
    models are scored on the same resampled rows, the shared sampling noise
    cancels and the interval reflects only the disagreement between them.

    Returns the observed difference, the bootstrap mean, a 95% percentile
    interval, and the fraction of resamples in which A beat B.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if labels is None:
        labels = sorted(set(y_true.tolist()))

    def macro_f1(idx, pred):
        return float(f1_score(y_true[idx], pred[idx], average="macro", labels=labels, zero_division=0))

    n = y_true.shape[0]
    all_idx = np.arange(n)
    observed = macro_f1(all_idx, pred_a) - macro_f1(all_idx, pred_b)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = macro_f1(idx, pred_a) - macro_f1(idx, pred_b)

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "observed_difference": observed,
        "bootstrap_mean": float(diffs.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "fraction_a_greater": float((diffs > 0).mean()),
        "n_boot": int(n_boot),
        "n_rows": int(n),
        "seed": int(seed),
    }
