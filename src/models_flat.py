import numpy as np

from src.evaluate import family_metrics, subclass_metrics
from src.features import SPARSE_TRACE_COLUMNS, add_missingness_indicators, build_preprocessor
from src.model_training import train_and_eval_models


def train_flat(
    train_df, eval_df, numeric_cols, categorical_cols,
    label_col="class_label", model_names=None, seed=42,
):
    sparse_present = [c for c in SPARSE_TRACE_COLUMNS if c in train_df.columns]

    train_df = add_missingness_indicators(train_df, sparse_present)
    eval_df = add_missingness_indicators(eval_df, sparse_present)
    indicator_cols = [f"{c}_is_missing" for c in sparse_present]

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols + indicator_cols, categorical_cols=categorical_cols
    )
    X_train = preprocessor.fit_transform(train_df)
    X_eval = preprocessor.transform(eval_df)

    y_train = train_df[label_col].values
    y_eval = eval_df[label_col].values

    models = train_and_eval_models(
        X_train, y_train, X_eval, y_eval, model_names=model_names, seed=seed
    )

    return {"class_true": y_eval, "models": models}


def flat_comparison_metrics(
    class_true, class_pred, family_true, family_pred,
    pq_hybrid_families=("Post-Quantum", "Hybrid"),
):
    """Derive the three flat-vs-hierarchical comparison metrics from one
    flat model's predictions.

    - flat_24class: the flat model's raw 24-way macro-F1 (no hierarchy).
    - flat_family_derived: the same predictions collapsed to family,
      comparable to Stage 1's family_metrics output.
    - flat_subtype_pq_hybrid: the same predictions restricted to rows whose
      TRUE family is PQ/Hybrid (never predicted family — a row the flat
      model mispredicts into a different family must still count here, the
      same way Stage 2 would have seen it), scored ONLY over the class
      labels that actually occur (in ground truth) within that subset. This
      label restriction is what makes the number comparable to Stage 2's
      macro-F1: Stage 2's model is trained exclusively on PQ/Hybrid rows, so
      it can never predict a class outside this same set, while the flat
      model (trained on all 24 classes) can and does — each such
      out-of-family prediction would otherwise enter the label union with
      zero true instances and drag macro-F1 down through a mechanism Stage 2
      is structurally immune to.
    """
    class_true = np.asarray(class_true)
    class_pred = np.asarray(class_pred)
    family_true = np.asarray(family_true)
    family_pred = np.asarray(family_pred)

    flat_24class = subclass_metrics(class_true, class_pred)
    flat_family_derived = family_metrics(family_true, family_pred)

    pq_hybrid_mask = np.isin(family_true, list(pq_hybrid_families))
    subtype_true = class_true[pq_hybrid_mask]
    subtype_pred = class_pred[pq_hybrid_mask]
    subtype_labels = sorted(set(subtype_true))
    flat_subtype_pq_hybrid = subclass_metrics(subtype_true, subtype_pred, labels=subtype_labels)

    return {
        "flat_24class": flat_24class,
        "flat_family_derived": flat_family_derived,
        "flat_subtype_pq_hybrid": flat_subtype_pq_hybrid,
    }
