from src.features import SPARSE_TRACE_COLUMNS, add_missingness_indicators, build_preprocessor
from src.model_training import train_and_eval_models

FAMILY_LABEL_COL = "attack_family"


def train_stage1(
    train_df, eval_df, numeric_cols, categorical_cols,
    label_col=FAMILY_LABEL_COL, model_names=None, seed=42,
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

    return {"family_true": y_eval, "models": models}
