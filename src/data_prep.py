from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def _decode(values):
    return [x.decode() if isinstance(x, bytes) else x for x in values]


def check_corpus_identity(
    csv_path,
    h5_path,
    key_cols=("vehicle_id", "timestamp", "session_id"),
    n=5000,
):
    key_cols = list(key_cols)
    csv_head = pd.read_csv(csv_path, usecols=key_cols, nrows=n)

    with h5py.File(h5_path, "r") as f:
        n_available = min(n, f[key_cols[0]].shape[0])
        h5_head = pd.DataFrame(
            {col: _decode(f[col][:n_available]) for col in key_cols}
        )

    n_compared = min(len(csv_head), len(h5_head))
    csv_head = csv_head[key_cols].iloc[:n_compared].astype(str).reset_index(drop=True)
    h5_head = h5_head[key_cols].iloc[:n_compared].astype(str).reset_index(drop=True)

    positional_match_rate = float((csv_head == h5_head).all(axis=1).mean())

    csv_keys = set(map(tuple, csv_head.values))
    h5_keys = set(map(tuple, h5_head.values))
    denom = max(len(csv_keys), 1)
    set_overlap_rate = float(len(csv_keys & h5_keys) / denom)

    return {
        "n_compared": n_compared,
        "positional_match_rate": positional_match_rate,
        "set_overlap_rate": set_overlap_rate,
        "same_corpus_same_order": positional_match_rate > 0.99,
        "same_corpus_reordered": set_overlap_rate > 0.99,
    }


def check_session_leakage(df, group_col="session_id"):
    counts = df[group_col].value_counts()
    multi_row_groups = counts[counts > 1]
    return {
        "n_unique_groups": int(counts.shape[0]),
        "n_rows": int(len(df)),
        "max_rows_per_group": int(counts.max()) if len(counts) else 0,
        "rows_in_multi_row_groups": int(multi_row_groups.sum()),
        "leakage_risk": bool(len(counts) and counts.max() > 1),
    }


def csv_to_parquet(csv_path, out_dir, chunksize=200_000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"

    writer = None
    schema = None
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema)
            else:
                table = table.cast(schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return out_path


def h5_to_parquet(h5_path, out_dir, chunksize=200_000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"

    writer = None
    schema = None
    with h5py.File(h5_path, "r") as f:
        columns = list(f.keys())
        n_rows = f[columns[0]].shape[0]
        try:
            for start in range(0, n_rows, chunksize):
                end = min(start + chunksize, n_rows)
                chunk = {}
                for col in columns:
                    values = f[col][start:end]
                    if values.dtype == object or values.dtype.kind == "S":
                        values = np.array(_decode(values))
                    chunk[col] = values
                df_chunk = pd.DataFrame(chunk)
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(out_path, schema)
                else:
                    table = table.cast(schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
    return out_path


def stratified_split(df, label_col="class_label", group_col=None, ratios=(0.7, 0.15, 0.15), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    train_ratio, val_ratio, test_ratio = ratios

    if group_col is None:
        train_df, temp_df = train_test_split(
            df, train_size=train_ratio, stratify=df[label_col], random_state=seed
        )
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, train_size=relative_val_ratio, stratify=temp_df[label_col], random_state=seed
        )
    else:
        n_splits_1 = max(round(1 / test_ratio), 2)
        splitter_1 = StratifiedGroupKFold(n_splits=n_splits_1, shuffle=True, random_state=seed)
        trainval_idx, test_idx = next(
            splitter_1.split(df, df[label_col], groups=df[group_col])
        )
        trainval_df = df.iloc[trainval_idx]
        test_df = df.iloc[test_idx]

        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        n_splits_2 = max(round(1 / relative_val_ratio), 2)
        splitter_2 = StratifiedGroupKFold(n_splits=n_splits_2, shuffle=True, random_state=seed)
        train_idx, val_idx = next(
            splitter_2.split(
                trainval_df, trainval_df[label_col], groups=trainval_df[group_col]
            )
        )
        train_df = trainval_df.iloc[train_idx]
        val_df = trainval_df.iloc[val_idx]

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
