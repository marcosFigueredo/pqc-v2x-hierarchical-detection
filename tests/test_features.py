import numpy as np
import pandas as pd

from src.features import (
    LEAKY_COLUMNS,
    SPARSE_TRACE_COLUMNS,
    add_missingness_indicators,
    audit_leakage,
    build_preprocessor,
    derive_family_label,
)


def test_leaky_columns_constant_contains_known_leaks():
    for col in ["label", "class_label", "risk_score", "synthetic_anomaly_score"]:
        assert col in LEAKY_COLUMNS


def test_sparse_trace_columns_constant_contains_known_sparse_fields():
    for col in ["pqc_public_key", "handshake_time", "cipher_suite"]:
        assert col in SPARSE_TRACE_COLUMNS


def test_audit_leakage_flags_separable_column():
    rng = np.random.default_rng(0)
    n = 2000
    is_attack = rng.integers(0, 2, size=n)
    leaky = is_attack * 0.5 + rng.normal(0, 0.01, size=n)
    safe = rng.normal(0, 1, size=n)
    df = pd.DataFrame({"is_attack": is_attack, "leaky_col": leaky, "safe_col": safe})

    result = audit_leakage(df, candidate_cols=["leaky_col", "safe_col"])

    leaky_row = result[result["column"] == "leaky_col"].iloc[0]
    safe_row = result[result["column"] == "safe_col"].iloc[0]
    assert leaky_row["likely_leakage"] == True
    assert safe_row["likely_leakage"] == False


def test_derive_family_label_covers_all_24_classes_with_no_nulls():
    class_labels = pd.Series(range(24))
    families = derive_family_label(class_labels)

    assert families.isna().sum() == 0
    assert families[0] == "Normal"
    assert set(families[1:11]) == {"Post-Quantum"}
    assert set(families[[11, 12, 13, 14, 19, 20, 21, 22, 23]]) == {"Hybrid"}
    assert set(families[15:19]) == {"Classical"}


def test_add_missingness_indicators():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
    result = add_missingness_indicators(df, ["a", "b"])
    assert list(result["a_is_missing"]) == [0, 1, 0]
    assert list(result["b_is_missing"]) == [0, 0, 1]
    assert "a" in result.columns and "b" in result.columns


def test_build_preprocessor_fits_and_transforms_mixed_types():
    df = pd.DataFrame({
        "num1": [1.0, None, 3.0, 4.0],
        "cat1": ["a", "b", None, "a"],
    })
    preprocessor = build_preprocessor(numeric_cols=["num1"], categorical_cols=["cat1"])
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 4
    assert not np.isnan(transformed).any()
