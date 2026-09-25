import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data_prep import check_corpus_identity, check_session_leakage, csv_to_parquet, h5_to_parquet, stratified_split


def _write_csv(path, df):
    df.to_csv(path, index=False)


def _write_h5(path, df):
    with h5py.File(path, "w") as f:
        for col in df.columns:
            values = df[col].values
            if pd.api.types.is_string_dtype(df[col]):
                f.create_dataset(col, data=np.asarray(values, dtype="S32"))
            else:
                f.create_dataset(col, data=values)


def test_check_corpus_identity_same_corpus(tmp_path):
    df = pd.DataFrame({
        "vehicle_id": [f"v{i}" for i in range(10)],
        "timestamp": [f"2026-01-01T00:00:{i:02d}" for i in range(10)],
        "session_id": [f"s{i}" for i in range(10)],
        "extra_h5_only": np.arange(10, dtype="float32"),
    })
    csv_path = tmp_path / "corpus.csv"
    h5_path = tmp_path / "corpus.h5"
    _write_csv(csv_path, df[["vehicle_id", "timestamp", "session_id"]])
    _write_h5(h5_path, df)

    result = check_corpus_identity(csv_path, h5_path, n=10)

    assert result["n_compared"] == 10
    assert result["positional_match_rate"] == 1.0
    assert result["set_overlap_rate"] == 1.0
    assert result["same_corpus_same_order"] is True
    assert result["same_corpus_reordered"] is True


def test_check_corpus_identity_different_corpus(tmp_path):
    df_csv = pd.DataFrame({
        "vehicle_id": [f"v{i}" for i in range(10)],
        "timestamp": [f"2026-01-01T00:00:{i:02d}" for i in range(10)],
        "session_id": [f"s{i}" for i in range(10)],
    })
    df_h5 = pd.DataFrame({
        "vehicle_id": [f"other{i}" for i in range(10)],
        "timestamp": [f"2099-01-01T00:00:{i:02d}" for i in range(10)],
        "session_id": [f"other_s{i}" for i in range(10)],
    })
    csv_path = tmp_path / "corpus.csv"
    h5_path = tmp_path / "corpus.h5"
    _write_csv(csv_path, df_csv)
    _write_h5(h5_path, df_h5)

    result = check_corpus_identity(csv_path, h5_path, n=10)

    assert result["positional_match_rate"] == 0.0
    assert result["set_overlap_rate"] == 0.0
    assert result["same_corpus_same_order"] is False
    assert result["same_corpus_reordered"] is False


def test_check_session_leakage_detects_repeats():
    df = pd.DataFrame({"session_id": ["a", "a", "b", "c", "c", "c"]})
    result = check_session_leakage(df)
    assert result["n_unique_groups"] == 3
    assert result["n_rows"] == 6
    assert result["max_rows_per_group"] == 3
    assert result["rows_in_multi_row_groups"] == 5
    assert result["leakage_risk"] is True


def test_check_session_leakage_no_repeats():
    df = pd.DataFrame({"session_id": ["a", "b", "c"]})
    result = check_session_leakage(df)
    assert result["max_rows_per_group"] == 1
    assert result["rows_in_multi_row_groups"] == 0
    assert result["leakage_risk"] is False


def test_csv_to_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({
        "a": range(5),
        "b": [f"x{i}" for i in range(5)],
        "c": [1.5, None, 2.5, None, 3.5],
    })
    csv_path = tmp_path / "in.csv"
    df.to_csv(csv_path, index=False)

    out_path = csv_to_parquet(csv_path, tmp_path / "out_csv", chunksize=2)

    result = pq.read_table(out_path).to_pandas()
    assert len(result) == 5
    assert list(result["a"]) == [0, 1, 2, 3, 4]
    assert list(result["b"]) == ["x0", "x1", "x2", "x3", "x4"]


def test_h5_to_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({
        "a": range(5),
        "b": [f"x{i}".encode() for i in range(5)],
    })
    h5_path = tmp_path / "in.h5"
    _write_h5(h5_path, df)

    out_path = h5_to_parquet(h5_path, tmp_path / "out_h5", chunksize=2)

    result = pq.read_table(out_path).to_pandas()
    assert len(result) == 5
    assert list(result["a"]) == [0, 1, 2, 3, 4]
    assert list(result["b"]) == ["x0", "x1", "x2", "x3", "x4"]


def test_stratified_split_plain_preserves_class_ratio():
    n_per_class = 100
    df = pd.DataFrame({
        "class_label": np.repeat([0, 1, 2, 3], n_per_class),
        "value": np.arange(4 * n_per_class),
    })
    train_df, val_df, test_df = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    for split_df in (train_df, val_df, test_df):
        counts = split_df["class_label"].value_counts(normalize=True)
        for cls in [0, 1, 2, 3]:
            assert abs(counts[cls] - 0.25) < 0.05


def test_stratified_split_grouped_keeps_groups_together():
    groups = np.repeat(np.arange(40), 5)
    labels = np.tile(np.repeat([0, 1], 5), 20)
    df = pd.DataFrame({
        "class_label": labels,
        "session_id": groups,
    })

    train_df, val_df, test_df = stratified_split(
        df, group_col="session_id", ratios=(0.7, 0.15, 0.15), seed=42
    )

    train_groups = set(train_df["session_id"])
    val_groups = set(val_df["session_id"])
    test_groups = set(test_df["session_id"])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
