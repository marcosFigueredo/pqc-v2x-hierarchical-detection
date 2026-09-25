import numpy as np

from src.features import SPARSE_TRACE_COLUMNS, add_missingness_indicators, build_preprocessor
from src.model_training import train_and_eval_models
from src.models_stage1 import FAMILY_LABEL_COL


def train_stage2(
    train_df,
    eval_df,
    numeric_cols,
    categorical_cols,
    family_col=FAMILY_LABEL_COL,
    subclass_col="class_label",
    pq_hybrid_families=("Post-Quantum", "Hybrid"),
    model_names=None,
    predict_df=None,
    seed=42,
):
    train_pq = train_df[train_df[family_col].isin(pq_hybrid_families)].reset_index(drop=True)
    eval_pq = eval_df[eval_df[family_col].isin(pq_hybrid_families)].reset_index(drop=True)

    sparse_present = [c for c in SPARSE_TRACE_COLUMNS if c in train_df.columns]

    train_pq = add_missingness_indicators(train_pq, sparse_present)
    eval_pq = add_missingness_indicators(eval_pq, sparse_present)
    indicator_cols = [f"{c}_is_missing" for c in sparse_present]

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols + indicator_cols, categorical_cols=categorical_cols
    )
    X_train = preprocessor.fit_transform(train_pq)
    X_eval = preprocessor.transform(eval_pq)

    y_train = train_pq[subclass_col].values
    y_eval = eval_pq[subclass_col].values

    models = train_and_eval_models(
        X_train, y_train, X_eval, y_eval, model_names=model_names, seed=seed
    )

    result = {"subclass_true": y_eval, "models": models}

    if predict_df is not None:
        # Deliberately NOT filtered by family_col — predict_df is the set of
        # rows Stage 1 *predicted* as PQ/Hybrid, which may include rows whose
        # true family is Normal/Classical (Stage 1 false positives) and may
        # be missing true PQ/Hybrid rows Stage 1 routed elsewhere (false
        # negatives). Filtering here would silently turn this back into the
        # oracle-conditioned evaluation this parameter exists to avoid.
        predict_df = predict_df.reset_index(drop=True)
        predict_df = add_missingness_indicators(predict_df, sparse_present)
        X_predict = preprocessor.transform(predict_df)
        result["routed_true"] = predict_df[subclass_col].values
        result["routed_predictions"] = {
            name: res["model"].predict(X_predict) for name, res in models.items()
        }

    return result


def routing_diagnostics(true_family, predicted_family, pq_hybrid_families=("Post-Quantum", "Hybrid")):
    true_family = np.asarray(true_family)
    predicted_family = np.asarray(predicted_family)

    truly_pq_hybrid = np.isin(true_family, list(pq_hybrid_families))
    routed = np.isin(predicted_family, list(pq_hybrid_families))

    routed_count = int(routed.sum())
    true_pq_hybrid_count = int(truly_pq_hybrid.sum())
    true_positive = int((routed & truly_pq_hybrid).sum())
    false_positive = int((routed & ~truly_pq_hybrid).sum())
    false_negative = int((~routed & truly_pq_hybrid).sum())

    return {
        "routed_count": routed_count,
        "true_pq_hybrid_count": true_pq_hybrid_count,
        "routed_and_truly_pq_hybrid": true_positive,
        "routed_but_not_truly_pq_hybrid": false_positive,
        "not_routed_but_truly_pq_hybrid": false_negative,
        "routing_precision": (true_positive / routed_count) if routed_count else 0.0,
        "routing_recall": (true_positive / true_pq_hybrid_count) if true_pq_hybrid_count else 0.0,
    }
