import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

LEAKY_COLUMNS = [
    "label",
    "class_label",
    "attack_type",
    "is_attack",
    "attack_family",
    "attack_severity",
    "detected_flag",
    "detection_method",
    "risk_score",
    "synthetic_anomaly_score",
    "risk_amp",
]

SPARSE_TRACE_COLUMNS = [
    "protocol_mismatch",
    "zone_mismatch_flag",
    "sequence_gap",
    "timestamp_offset",
    "message_frequency",
    "source_entropy",
    "handshake_time",
    "cipher_suite",
    "encryption_time",
    "block_mode",
    "sig_generation_time",
    "sig_verification_time",
    "pqc_public_key",
    "pqc_ciphertext",
    "pqc_signature",
]

# Class -> family mapping straight from the spec's attack taxonomy (README's
# "Paper reporting family" column). class_label is 100% populated in both
# releases; the raw attack_family column is not (see docstring above), so
# this is the only reliable source of the 4-class family label.
CLASS_LABEL_TO_FAMILY = {
    0: "Normal",
    **{c: "Post-Quantum" for c in range(1, 11)},
    **{c: "Hybrid" for c in (11, 12, 13, 14)},
    **{c: "Classical" for c in (15, 16, 17, 18)},
    **{c: "Hybrid" for c in (19, 20, 21, 22, 23)},
}


def derive_family_label(class_label_series):
    return class_label_series.map(CLASS_LABEL_TO_FAMILY)


def audit_leakage(df, candidate_cols, target_col="is_attack", auc_threshold=0.97):
    y = df[target_col].astype(int)
    rows = []
    for col in candidate_cols:
        values = df[col]
        if not pd.api.types.is_numeric_dtype(values):
            continue
        filled = values.fillna(values.median())
        if filled.nunique() < 2:
            auc = 0.5
        else:
            raw_auc = roc_auc_score(y, filled)
            auc = max(raw_auc, 1 - raw_auc)
        rows.append(
            {
                "column": col,
                "auc_vs_target": auc,
                "likely_leakage": auc >= auc_threshold,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("auc_vs_target", ascending=False)
        .reset_index(drop=True)
    )


def add_missingness_indicators(df, cols):
    df = df.copy()
    for col in cols:
        df[f"{col}_is_missing"] = df[col].isna().astype(int)
    return df


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            # Amendment 2 (2026-08-27, C2 implementation fix, discovered by the
            # smoke test): the H5 corpus's protocol_mismatch/zone_mismatch_flag
            # columns come back from Parquet as plain numpy bool (no NaN in that
            # release), while the rest of the categorical columns come back as
            # pandas' pyarrow-backed string dtype (this repo's pandas 3.0). A
            # ColumnTransformer sub-frame mixing those two dtypes fails
            # SimpleImputer's numeric-array validation ("could not convert
            # string to float") before it ever reaches the constant-fill step.
            # Casting to plain object dtype first — a no-op for the
            # already-object dtype columns from the CSV corpus, which never hit
            # this — makes every categorical column's dtype uniform regardless
            # of source corpus, matching how this pipeline already behaved for
            # every categorical column before the sparse trace/bool columns
            # were added.
            ("cast", FunctionTransformer(lambda X: X.astype("object"))),
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
