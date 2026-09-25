# PQC Hierarchical Detection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a leakage-safe, two-stage (family → PQ/Hybrid subclass) ML pipeline over both PQ-V2X releases (55-col CSV and 65-col H5) to test the three hypotheses in the spec, on CPU-only hardware.

**Architecture:** `src/data_prep.py` validates and converts both raw sources to Parquet and produces train/val/test splits; `src/features.py` applies the leakage-safe feature policy; `src/model_training.py` provides one shared 4-model (LightGBM/LogReg/RandomForest/MLP) trainer used by `src/models_stage1.py` (family) and `src/models_stage2.py` (PQ/Hybrid subclass); `src/evaluate.py` computes all metrics including the CSV-vs-H5 and real-vs-synthetic comparisons; `src/run_pipeline.py` orchestrates a full run on real data.

**Tech Stack:** Python 3.11, pandas, pyarrow, h5py, scikit-learn, lightgbm, numpy, pytest.

**Spec:** [docs/superpowers/specs/2026-08-26-pqc-hierarchical-detection-design.md](../specs/2026-08-26-pqc-hierarchical-detection-design.md)

## Global Constraints

- Fixed random seed `42` used in every split, model, and sampling operation.
- Leaky columns excluded from all model inputs: `label`, `class_label`, `attack_type`, `is_attack`, `attack_family`, `attack_severity`, `detected_flag`, `detection_method`, `risk_score`, `synthetic_anomaly_score`.
- Sparse crypto-trace-detail columns (`protocol_mismatch`, `zone_mismatch_flag`, `sequence_gap`, `timestamp_offset`, `message_frequency`, `source_entropy`, `handshake_time`, `cipher_suite`, `encryption_time`, `block_mode`, `sig_generation_time`, `sig_verification_time`, `pqc_public_key`, `pqc_ciphertext`, `pqc_signature`) get a `{col}_is_missing` indicator, never dropped, never mean/mode-imputed as if missing-at-random.
- Split: 70/15/15 train/val/test, stratified by `class_label`, **group-stratified by `session_id`** (Task 4 confirmed session-level row repetition against the real data — `leakage_risk: true`, max 5 rows/session in the first 500k CSV rows — so plain row-random stratification is not safe; see Task 6's `stratified_split(group_col=...)`).
- CPU-only: no GPU device flags anywhere.
- Every stage trains the same 4 models: LightGBM (primary), Logistic Regression, Random Forest, MLP (baselines).
- Raw files (`pq_v2x_realistic.csv`, `pq_v2x_dataset.h5`) live at `G:\My Drive\UNEB\PPGMSB\MarcosProducaoCientifica\2026\quantumStudies\quantumTest` (the original Google-Drive-synced folder) — never moved or copied, only read from. The code repo itself lives locally at `C:\Users\marco\dev\quantumTest` (moved off Drive during this plan's setup after repeated git lock corruption on the Drive path — see the SDD ledger's Setup section). Only `data/processed/` (Parquet, gitignored) and `results/` (small metrics/JSON, versioned) are written, both local.
- Paper writing is out of scope for this plan.

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/__init__.py`
- Create: `data/processed/.gitkeep`
- Create: `results/.gitkeep`

**Interfaces:**
- Produces: `src.config.DATA_SOURCE_DIR`, `src.config.RAW_CSV_PATH`, `src.config.RAW_H5_PATH`, `src.config.PROCESSED_DIR`, `src.config.RESULTS_DIR`, `src.config.SEED` (all `pathlib.Path` except `SEED: int`) — every later task imports these instead of hardcoding paths.

- [ ] **Step 1: Create `requirements.txt`**

```text
pandas>=2.0
pyarrow>=14.0
h5py>=3.9
scikit-learn>=1.3
lightgbm>=4.0
numpy>=1.24
pytest>=7.4
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Create `src/__init__.py` and `tests/__init__.py`** (both empty files)

- [ ] **Step 4: Create `src/config.py`**

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The raw dataset files are too large to live in the code repo and stay on
# the original Google-Drive-synced folder (the repo itself was moved off
# Drive to a local path after repeated git lock corruption there — see the
# SDD ledger). This is a single-machine research repo, so a hardcoded
# absolute path is the simplest correct thing; update it if the data moves.
DATA_SOURCE_DIR = Path(
    r"G:\My Drive\UNEB\PPGMSB\MarcosProducaoCientifica\2026\quantumStudies\quantumTest"
)

RAW_CSV_PATH = DATA_SOURCE_DIR / "pq_v2x_realistic.csv"
RAW_H5_PATH = DATA_SOURCE_DIR / "pq_v2x_dataset.h5"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_CORPUS_DIR = PROCESSED_DIR / "csv_corpus"
H5_CORPUS_DIR = PROCESSED_DIR / "h5_corpus"

RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42
```

- [ ] **Step 5: Create placeholder directories**

```bash
mkdir -p data/processed results
touch data/processed/.gitkeep results/.gitkeep
```

- [ ] **Step 6: Install dependencies and verify imports**

```bash
pip install -r requirements.txt
python -c "from src.config import RAW_CSV_PATH, SEED; print(RAW_CSV_PATH, SEED)"
```

Expected: prints the CSV path and `42` with no import errors.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pyproject.toml src/__init__.py src/config.py tests/__init__.py data/processed/.gitkeep results/.gitkeep
git commit -m "chore: project scaffolding for PQC detection pipeline"
```

---

## Task 2: Corpus identity check (CSV vs H5)

**Files:**
- Create: `src/data_prep.py`
- Test: `tests/test_data_prep.py`

**Interfaces:**
- Consumes: nothing from earlier tasks besides `src.config`.
- Produces: `check_corpus_identity(csv_path, h5_path, key_cols=("vehicle_id", "timestamp", "session_id"), n=5000) -> dict` with keys `n_compared`, `positional_match_rate`, `set_overlap_rate`, `same_corpus_same_order`, `same_corpus_reordered` (all Python floats/bools) — Task 4 calls this against the real files.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_prep.py
import h5py
import numpy as np
import pandas as pd

from src.data_prep import check_corpus_identity


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_data_prep.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError: cannot import name 'check_corpus_identity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data_prep.py
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


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
    csv_head = csv_head.iloc[:n_compared].astype(str).reset_index(drop=True)
    h5_head = h5_head.iloc[:n_compared].astype(str).reset_index(drop=True)

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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_data_prep.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_prep.py tests/test_data_prep.py
git commit -m "feat: add CSV/H5 corpus identity check"
```

---

## Task 3: Session-level leakage check

**Files:**
- Modify: `src/data_prep.py`
- Test: `tests/test_data_prep.py`

**Interfaces:**
- Produces: `check_session_leakage(df, group_col="session_id") -> dict` with keys `n_unique_groups`, `n_rows`, `max_rows_per_group`, `rows_in_multi_row_groups`, `leakage_risk` (bool) — Task 4 calls this against the real files; Task 6's split uses `leakage_risk` to decide whether to pass `group_col`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_data_prep.py
import pandas as pd

from src.data_prep import check_session_leakage


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_data_prep.py -v -k session_leakage
```

Expected: FAIL with `ImportError: cannot import name 'check_session_leakage'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/data_prep.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_data_prep.py -v -k session_leakage
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_prep.py tests/test_data_prep.py
git commit -m "feat: add session-level leakage check"
```

---

## Task 4: Run identity/leakage checks against real data (investigative)

This task has no unit test — it runs Task 2/3's functions against the real 2.8GB CSV and 5.9GB H5 files to resolve Open Technical Questions 1 and 2 from the spec, and its findings decide the split strategy used in Task 6/13.

**Files:**
- Create: `scripts/check_real_data.py`
- Modify: `docs/superpowers/specs/2026-08-26-pqc-hierarchical-detection-design.md` (record findings)

- [ ] **Step 1: Write the investigative script**

```python
# scripts/check_real_data.py
import json

import pandas as pd

from src.config import RAW_CSV_PATH, RAW_H5_PATH
from src.data_prep import check_corpus_identity, check_session_leakage

if __name__ == "__main__":
    identity = check_corpus_identity(RAW_CSV_PATH, RAW_H5_PATH, n=20000)
    print("Corpus identity check:")
    print(json.dumps(identity, indent=2))

    csv_sample = pd.read_csv(RAW_CSV_PATH, usecols=["session_id"], nrows=500_000)
    session_leakage = check_session_leakage(csv_sample)
    print("\nSession leakage check (first 500k CSV rows):")
    print(json.dumps(session_leakage, indent=2))
```

- [ ] **Step 2: Run it against the real files**

```bash
python scripts/check_real_data.py
```

Expected runtime: a few minutes (H5 random-ish access + CSV partial read). Note the printed JSON — both dicts are needed for the next step.

- [ ] **Step 3: Record the findings in the spec's "Open Technical Questions" section**

Open `docs/superpowers/specs/2026-08-26-pqc-hierarchical-detection-design.md` and replace the text of Open Technical Questions 1 and 2 with the actual measured values from Step 2's output (e.g. "Resolved 2026-XX-XX: positional_match_rate=..., set_overlap_rate=..., same_corpus_reordered=... → CSV and H5 are/are not the same underlying corpus" and "Resolved: max_rows_per_group=..., leakage_risk=... → split must/must not be grouped by session_id"). Do not guess these numbers — copy them from the actual Step 2 output.

- [ ] **Step 4: Commit**

```bash
git add scripts/check_real_data.py docs/superpowers/specs/2026-08-26-pqc-hierarchical-detection-design.md
git commit -m "docs: resolve corpus-identity and session-leakage open questions with real-data findings"
```

---

## Task 5: CSV/H5 → Parquet conversion

**Files:**
- Modify: `src/data_prep.py`
- Test: `tests/test_data_prep.py`

**Interfaces:**
- Produces: `csv_to_parquet(csv_path, out_dir, chunksize=200_000) -> Path` and `h5_to_parquet(h5_path, out_dir, chunksize=200_000) -> Path`, both writing a single `data.parquet` file inside `out_dir` and returning its path — Task 13's orchestration script calls both against the real files.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_data_prep.py
import pandas as pd
import pyarrow.parquet as pq

from src.data_prep import csv_to_parquet, h5_to_parquet


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_data_prep.py -v -k parquet
```

Expected: FAIL with `ImportError: cannot import name 'csv_to_parquet'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/data_prep.py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_data_prep.py -v -k parquet
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_prep.py tests/test_data_prep.py
git commit -m "feat: add chunked CSV/H5 to Parquet conversion"
```

---

## Task 6: Stratified (optionally grouped) split

**Files:**
- Modify: `src/data_prep.py`
- Test: `tests/test_data_prep.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `stratified_split(df, label_col="class_label", group_col=None, ratios=(0.7, 0.15, 0.15), seed=42) -> (train_df, val_df, test_df)` — Task 13's orchestration calls this with `group_col="session_id"` if Task 4 found session leakage, else `group_col=None`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_data_prep.py
import numpy as np
import pandas as pd

from src.data_prep import stratified_split


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_data_prep.py -v -k stratified_split
```

Expected: FAIL with `ImportError: cannot import name 'stratified_split'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/data_prep.py
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_data_prep.py -v -k stratified_split
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_prep.py tests/test_data_prep.py
git commit -m "feat: add stratified and group-stratified train/val/test split"
```

---

## Task 7: Leakage audit utility + feature-policy constants

**Files:**
- Create: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `LEAKY_COLUMNS: list[str]`, `SPARSE_TRACE_COLUMNS: list[str]`, `CLASS_LABEL_TO_FAMILY: dict[int, str]`, `derive_family_label(class_label_series) -> pd.Series`, `audit_leakage(df, candidate_cols, target_col="is_attack", auc_threshold=0.97) -> pd.DataFrame` (columns: `column`, `auc_vs_target`, `likely_leakage`) — Task 4-equivalent audit for the H5-only derived fields (Open Technical Question 3) calls this directly against real data in Task 8; Task 9's feature-matrix builder imports `LEAKY_COLUMNS`/`SPARSE_TRACE_COLUMNS`; Task 14's orchestration calls `derive_family_label` instead of trusting the raw `attack_family` column.

**Why `derive_family_label` exists:** verified against the real CSV this session — the raw `attack_family` column is NaN for every `is_attack=False` (Normal) row and for ~4% of attack rows too (only ~96% notna when `is_attack=True`), so it is NOT a clean 4-class label despite the README's family table implying one. `class_label` is 100% populated (confirmed this session) and is the reliable source; `derive_family_label` maps it to the same 4 families the README reports (Normal/Post-Quantum/Hybrid/Classical) using the exact class→family table from the spec's attack taxonomy.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import numpy as np
import pandas as pd

from src.features import (
    LEAKY_COLUMNS,
    SPARSE_TRACE_COLUMNS,
    audit_leakage,
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_features.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.features'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/features.py
import pandas as pd
from sklearn.metrics import roc_auc_score

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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_features.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add leakage audit utility, feature-policy constants, and family-label derivation"
```

---

## Task 8: Run H5-derived-field leakage audit against real data (investigative)

**Files:**
- Create: `scripts/audit_h5_features.py`
- Modify: `src/features.py` (extend `LEAKY_COLUMNS` if the audit finds new leaks)
- Modify: `docs/superpowers/specs/2026-08-26-pqc-hierarchical-detection-design.md` (record findings)

- [ ] **Step 1: Write the investigative script**

```python
# scripts/audit_h5_features.py
import h5py
import numpy as np
import pandas as pd

from src.config import RAW_H5_PATH
from src.features import audit_leakage

H5_ONLY_DERIVED_FIELDS = [
    "ZoneRisk_tunnel",
    "cos_timestamp",
    "sin_timestamp",
    "decryption_time",
    "enc_dec_ratio",
    "latency_ber_ratio",
    "risk_amp",
    "size_anomaly",
    "snr_packet_loss_ratio",
]

if __name__ == "__main__":
    n = 300_000
    with h5py.File(RAW_H5_PATH, "r") as f:
        data = {col: f[col][:n] for col in H5_ONLY_DERIVED_FIELDS}
        data["is_attack"] = f["is_attack"][:n].astype(int)
    df = pd.DataFrame(data)

    result = audit_leakage(df, candidate_cols=H5_ONLY_DERIVED_FIELDS)
    print(result.to_string(index=False))
```

- [ ] **Step 2: Run it against the real H5 file**

```bash
python scripts/audit_h5_features.py
```

Expected runtime: under a minute (only 9 columns × 300k rows read).

- [ ] **Step 3: Update `LEAKY_COLUMNS` in `src/features.py` with any field flagged `likely_leakage == True`**

Add each flagged column name to the `LEAKY_COLUMNS` list literal.

- [ ] **Step 4: Record findings in the spec**

Open `docs/superpowers/specs/2026-08-26-pqc-hierarchical-detection-design.md` and replace Open Technical Question 3's text with the actual audit table from Step 2 and which columns (if any) were added to `LEAKY_COLUMNS`.

- [ ] **Step 5: Re-run feature tests to confirm nothing broke**

```bash
python -m pytest tests/test_features.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit_h5_features.py src/features.py docs/superpowers/specs/2026-08-26-pqc-hierarchical-detection-design.md
git commit -m "docs: resolve H5 derived-field leakage audit with real-data findings"
```

---

## Task 9: Missingness indicators + preprocessing pipeline

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `add_missingness_indicators(df, cols) -> pd.DataFrame` (adds `{col}_is_missing` int columns) and `build_preprocessor(numeric_cols, categorical_cols) -> sklearn.compose.ColumnTransformer` — Tasks 10/11's `build_feature_matrix`-equivalent code in `models_stage1.py`/`models_stage2.py` calls both.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_features.py
import numpy as np
import pandas as pd

from src.features import add_missingness_indicators, build_preprocessor


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_features.py -v -k "missingness or preprocessor"
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/features.py
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_features.py -v -k "missingness or preprocessor"
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add missingness indicators and shared preprocessing pipeline"
```

---

## Task 10: Shared 4-model trainer

**Files:**
- Create: `src/model_training.py`
- Test: `tests/test_model_training.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `get_models(seed=42) -> dict[str, estimator]` (keys: `"lightgbm"`, `"logreg"`, `"random_forest"`, `"mlp"`) and `train_and_eval_models(X_train, y_train, X_val, y_val, seed=42) -> dict[str, dict]` where each value has keys `"model"` (fitted estimator) and `"val_predictions"` (`np.ndarray`) — Tasks 11/12 call `train_and_eval_models` directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_training.py
import numpy as np
from sklearn.datasets import make_classification

from src.model_training import get_models, train_and_eval_models


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
        assert res["val_predictions"].shape[0] == X_val.shape[0]
        assert set(np.unique(res["val_predictions"])).issubset({0, 1, 2})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_model_training.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.model_training'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/model_training.py
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


def train_and_eval_models(X_train, y_train, X_val, y_val, seed=42):
    sample_weight = compute_sample_weight("balanced", y_train)
    results = {}
    for name, model in get_models(seed=seed).items():
        if name == "mlp":
            # sklearn's MLPClassifier.fit has no sample_weight parameter,
            # so it trains unweighted; documented limitation, baseline only.
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        val_predictions = model.predict(X_val)
        results[name] = {"model": model, "val_predictions": val_predictions}
    return results
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_model_training.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/model_training.py tests/test_model_training.py
git commit -m "feat: add shared 4-model (LightGBM/LogReg/RF/MLP) trainer"
```

---

## Task 11: Stage 1 — family classifier

**Files:**
- Create: `src/models_stage1.py`
- Test: `tests/test_models_stage1.py`

**Interfaces:**
- Consumes: `src.features.LEAKY_COLUMNS`, `SPARSE_TRACE_COLUMNS`, `add_missingness_indicators`, `build_preprocessor`; `src.model_training.train_and_eval_models`.
- Produces: `FAMILY_LABEL_COL = "attack_family"` (fallback family derived from `class_label` — see Step 3) and `train_stage1(train_df, val_df, numeric_cols, categorical_cols) -> dict` with keys `"family_true"` (`np.ndarray` of val family labels) and `"models"` (the `train_and_eval_models` return value) — Task 13's orchestration and Task 12's evaluation call this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_stage1.py
import numpy as np
import pandas as pd

from src.models_stage1 import train_stage1


def _synthetic_df(n):
    rng = np.random.default_rng(0)
    class_label = rng.integers(0, 4, size=n)
    family_map = {0: "Normal", 1: "Post-Quantum", 2: "Hybrid", 3: "Classical"}
    return pd.DataFrame(
        {
            "class_label": class_label,
            "family": [family_map[c] for c in class_label],
            "num_feat": class_label + rng.normal(0, 0.5, size=n),
            "cat_feat": rng.choice(["urban", "rural"], size=n),
        }
    )


def test_train_stage1_returns_predictions_for_all_val_rows():
    train_df = _synthetic_df(300)
    val_df = _synthetic_df(100)

    result = train_stage1(
        train_df, val_df, numeric_cols=["num_feat"], categorical_cols=["cat_feat"],
        label_col="family",
    )

    assert len(result["family_true"]) == len(val_df)
    for name, res in result["models"].items():
        assert res["val_predictions"].shape[0] == len(val_df)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_models_stage1.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.models_stage1'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/models_stage1.py
from src.features import SPARSE_TRACE_COLUMNS, add_missingness_indicators, build_preprocessor
from src.model_training import train_and_eval_models

FAMILY_LABEL_COL = "attack_family"


def train_stage1(train_df, val_df, numeric_cols, categorical_cols, label_col=FAMILY_LABEL_COL):
    sparse_present = [c for c in SPARSE_TRACE_COLUMNS if c in numeric_cols + categorical_cols]

    train_df = add_missingness_indicators(train_df, sparse_present)
    val_df = add_missingness_indicators(val_df, sparse_present)
    indicator_cols = [f"{c}_is_missing" for c in sparse_present]

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols + indicator_cols, categorical_cols=categorical_cols
    )
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)

    y_train = train_df[label_col].values
    y_val = val_df[label_col].values

    models = train_and_eval_models(X_train, y_train, X_val, y_val)

    return {"family_true": y_val, "models": models}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_models_stage1.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models_stage1.py tests/test_models_stage1.py
git commit -m "feat: add Stage 1 family classifier pipeline"
```

---

## Task 12: Stage 2 — PQ/Hybrid subclass classifier

**Files:**
- Create: `src/models_stage2.py`
- Test: `tests/test_models_stage2.py`

**Interfaces:**
- Consumes: same as Task 11, plus `src.models_stage1.FAMILY_LABEL_COL`.
- Produces: `train_stage2(train_df, val_df, numeric_cols, categorical_cols, family_col=FAMILY_LABEL_COL, subclass_col="class_label", pq_hybrid_families=("Post-Quantum", "Hybrid")) -> dict` with keys `"subclass_true"` and `"models"` — trains only on rows whose ground-truth family is PQ or Hybrid; Task 13's orchestration and evaluation call this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_stage2.py
import numpy as np
import pandas as pd

from src.models_stage2 import train_stage2


def _synthetic_df(n):
    rng = np.random.default_rng(1)
    family = rng.choice(["Normal", "Post-Quantum", "Hybrid", "Classical"], size=n)
    subclass = np.where(
        family == "Post-Quantum", rng.integers(1, 5, size=n),
        np.where(family == "Hybrid", rng.integers(11, 15, size=n), 0),
    )
    return pd.DataFrame(
        {
            "attack_family": family,
            "class_label": subclass,
            "num_feat": subclass + rng.normal(0, 0.5, size=n),
            "cat_feat": rng.choice(["urban", "rural"], size=n),
        }
    )


def test_train_stage2_only_uses_pq_hybrid_rows():
    train_df = _synthetic_df(400)
    val_df = _synthetic_df(150)

    result = train_stage2(train_df, val_df, numeric_cols=["num_feat"], categorical_cols=["cat_feat"])

    expected_val_n = (val_df["attack_family"].isin(["Post-Quantum", "Hybrid"])).sum()
    assert len(result["subclass_true"]) == expected_val_n
    for name, res in result["models"].items():
        assert res["val_predictions"].shape[0] == expected_val_n
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_models_stage2.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.models_stage2'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/models_stage2.py
from src.features import SPARSE_TRACE_COLUMNS, add_missingness_indicators, build_preprocessor
from src.model_training import train_and_eval_models
from src.models_stage1 import FAMILY_LABEL_COL


def train_stage2(
    train_df,
    val_df,
    numeric_cols,
    categorical_cols,
    family_col=FAMILY_LABEL_COL,
    subclass_col="class_label",
    pq_hybrid_families=("Post-Quantum", "Hybrid"),
):
    train_pq = train_df[train_df[family_col].isin(pq_hybrid_families)].reset_index(drop=True)
    val_pq = val_df[val_df[family_col].isin(pq_hybrid_families)].reset_index(drop=True)

    sparse_present = [c for c in SPARSE_TRACE_COLUMNS if c in numeric_cols + categorical_cols]

    train_pq = add_missingness_indicators(train_pq, sparse_present)
    val_pq = add_missingness_indicators(val_pq, sparse_present)
    indicator_cols = [f"{c}_is_missing" for c in sparse_present]

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols + indicator_cols, categorical_cols=categorical_cols
    )
    X_train = preprocessor.fit_transform(train_pq)
    X_val = preprocessor.transform(val_pq)

    y_train = train_pq[subclass_col].values
    y_val = val_pq[subclass_col].values

    models = train_and_eval_models(X_train, y_train, X_val, y_val)

    return {"subclass_true": y_val, "models": models}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_models_stage2.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models_stage2.py tests/test_models_stage2.py
git commit -m "feat: add Stage 2 PQ/Hybrid subclass classifier pipeline"
```

---

## Task 13: Evaluation metrics

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: nothing new (works on plain arrays/dataframes).
- Produces: `family_metrics(y_true, y_pred) -> dict` (`accuracy`, `macro_f1`, `confusion_matrix` as nested list), `subclass_metrics(y_true, y_pred) -> dict` (`macro_f1`, `per_class` dict of precision/recall/f1), `compare_real_synthetic(df, y_true_col, y_pred_col, is_real_trace_col="is_real_trace") -> dict` (keys `"real"`, `"synthetic"`, each a `family_metrics`-shaped dict) — Task 14's orchestration calls all three.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np
import pandas as pd

from src.evaluate import compare_real_synthetic, family_metrics, subclass_metrics


def test_family_metrics_perfect_predictions():
    y_true = np.array(["Normal", "Post-Quantum", "Hybrid", "Classical"])
    y_pred = y_true.copy()
    result = family_metrics(y_true, y_pred)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert len(result["confusion_matrix"]) == 4


def test_subclass_metrics_reports_per_class():
    y_true = np.array([1, 1, 2, 2, 3])
    y_pred = np.array([1, 2, 2, 2, 3])
    result = subclass_metrics(y_true, y_pred)
    assert set(result["per_class"].keys()) == {1, 2, 3}
    assert 0.0 <= result["macro_f1"] <= 1.0


def test_compare_real_synthetic_splits_by_flag():
    df = pd.DataFrame({
        "y_true": ["Normal", "Normal", "Hybrid", "Hybrid"],
        "y_pred": ["Normal", "Hybrid", "Hybrid", "Hybrid"],
        "is_real_trace": [True, True, False, False],
    })
    result = compare_real_synthetic(df, "y_true", "y_pred")
    assert result["real"]["accuracy"] == 0.5
    assert result["synthetic"]["accuracy"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_evaluate.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluate.py
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


def subclass_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels))
    per_class = {
        label: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for label, p, r, f in zip(labels, precision, recall, f1)
    }
    return {"macro_f1": macro_f1, "per_class": per_class}


def compare_real_synthetic(df, y_true_col, y_pred_col, is_real_trace_col="is_real_trace"):
    real_df = df[df[is_real_trace_col] == True]
    synthetic_df = df[df[is_real_trace_col] == False]
    return {
        "real": family_metrics(real_df[y_true_col], real_df[y_pred_col]),
        "synthetic": family_metrics(synthetic_df[y_true_col], synthetic_df[y_pred_col]),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_evaluate.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: add family/subclass metrics and real-vs-synthetic comparison"
```

---

## Task 14: End-to-end orchestration on real data

**Amendment (2026-08-26, after a real-data smoke test on this task):** a 40,000-row
smoke test measured Stage 1 (all 4 models) at 960.0s and Stage 2 at 324.9s —
linearly extrapolated to the real ~3.5M-row training set, that's roughly
47+ hours for the CSV corpus alone, multi-day for both corpora. Not
achievable as originally specified. The user chose (of four options
presented): run the full 5M-row data with **LightGBM only** — the spec's
three hypotheses (hierarchical structure, CSV-vs-H5, real-vs-synthetic)
are measured entirely off LightGBM's predictions already (`best_stage1_preds
= stage1["models"]["lightgbm"]["val_predictions"]`), so this changes
nothing about what's actually measured; it only stops RandomForest/MLP
from being trained (and silently discarded) at full scale. RF/LogReg/MLP
remain trained and validated at the unit-test/smoke-test scale from Tasks
10-12 — this is a documented limitation for the real run, not a removal of
those baselines from the codebase.

This requires adding an optional, backward-compatible `model_names` filter
through `train_and_eval_models` (Task 10's `src/model_training.py`),
`train_stage1` (Task 11's `src/models_stage1.py`), and `train_stage2`
(Task 12's `src/models_stage2.py`) — each already-implemented, already-
reviewed function keeps its exact current behavior when the new parameter
is omitted (defaults to `None` = train all 4 models, so every existing
test in `tests/test_model_training.py`, `tests/test_models_stage1.py`,
`tests/test_models_stage2.py` keeps passing unmodified), and only restricts
training to the given subset when it's passed. This task modifies those
three files in addition to creating `src/run_pipeline.py`.

**Files:**
- Create: `src/run_pipeline.py`
- Modify: `src/model_training.py` (add optional `model_names` parameter to `train_and_eval_models`)
- Modify: `src/models_stage1.py` (add optional `model_names` parameter to `train_stage1`, passed through)
- Modify: `src/models_stage2.py` (add optional `model_names` parameter to `train_stage2`, passed through)

**Interfaces:**
- Consumes: everything from Tasks 1–13.
- Produces: `results/<corpus>_stage1_metrics.json`, `results/<corpus>_stage2_metrics.json`, `results/<corpus>_real_vs_synthetic.json` for `corpus in {"csv", "h5"}`, plus `results/csv_vs_h5_comparison.json`.

This task has no unit test for the orchestration script itself — it is the
real run over the full 5M-row files (2.8GB CSV, 5.9GB H5) using every
function built in Tasks 1–13, so correctness is judged by inspecting the
JSON outputs, not by a synthetic-data assertion. The `model_names`
additions to Tasks 10-12's files DO need a quick regression check: re-run
`tests/test_model_training.py`, `tests/test_models_stage1.py`,
`tests/test_models_stage2.py` (5 tests total across the three files) after
making the change, to confirm the default (`model_names=None`) behavior is
unchanged.

- [ ] **Step 0: Add the `model_names` filter to Tasks 10-12's files**

In `src/model_training.py`, change `train_and_eval_models`'s signature and
body to:

```python
def train_and_eval_models(X_train, y_train, X_val, y_val, seed=42, model_names=None):
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
        val_predictions = model.predict(X_val)
        results[name] = {"model": model, "val_predictions": val_predictions}
    return results
```

In `src/models_stage1.py`, add `model_names=None` to `train_stage1`'s
signature and forward it: `train_and_eval_models(X_train, y_train, X_val,
y_val, model_names=model_names)`.

In `src/models_stage2.py`, add `model_names=None` to `train_stage2`'s
signature and forward it the same way.

Re-run the three existing test files (5 tests) and confirm they still
pass unmodified — the new parameter must not change default behavior.

- [ ] **Step 1: Write the orchestration script**

```python
# src/run_pipeline.py
import json

import numpy as np
import pandas as pd

from src.config import CSV_CORPUS_DIR, H5_CORPUS_DIR, RAW_CSV_PATH, RAW_H5_PATH, RESULTS_DIR, SEED
from src.data_prep import csv_to_parquet, h5_to_parquet, stratified_split
from src.evaluate import compare_real_synthetic, family_metrics, subclass_metrics
from src.features import LEAKY_COLUMNS, derive_family_label
from src.models_stage1 import FAMILY_LABEL_COL, train_stage1
from src.models_stage2 import train_stage2

# Task 4 confirmed session-level leakage in the real data (max_rows_per_group=5,
# leakage_risk=true, first 500k CSV rows — see the spec's Open Technical
# Questions). Grouped-stratified split by session_id is mandatory.
SPLIT_GROUP_COL = "session_id"

# Amendment (2026-08-27): direct inspection of the real H5 file found that
# its session_id column is a constant 0.0 for every row (checked 200,000
# rows) — a placeholder/unpopulated field in that release, unlike the
# CSV's real per-observation UUID. Task 4's session-leakage check only
# ever ran against the CSV, never the H5, so this went unnoticed until a
# grouped split on the H5 corpus crashed with
# "n_splits=6 greater than the number of samples: n_samples=0" (every row
# collapsed into a single degenerate group). There is no usable grouping
# signal for H5, so it falls back to plain stratified splitting; the CSV
# corpus keeps the grouped split.
SPLIT_GROUP_COL_BY_CORPUS = {"csv": "session_id", "h5": None}

# A 40,000-row smoke test measured Stage 1 (all 4 models) at 960.0s and
# Stage 2 at 324.9s — linearly extrapolated to the real ~3.5M-row training
# set, that's 47+ hours for the CSV corpus alone. Per the user's decision,
# the real full-scale run trains LightGBM only (RandomForest/LogReg/MLP
# remain trained and validated at unit-test/smoke-test scale from Tasks
# 10-12, not at the real 5M-row scale). The spec's three hypotheses only
# ever read LightGBM's predictions, so this changes nothing about what's
# measured.
REAL_RUN_MODEL_NAMES = ["lightgbm"]

# Columns present in both corpora used as model inputs (everything else in
# each corpus stays out per LEAKY_COLUMNS / is only in one corpus).
NUMERIC_COLS = [
    "speed", "acceleration", "direction", "snr", "packet_loss", "latency",
    "ber", "KeyReuseCount", "EncapTimeDeviation", "EntropyDeviation",
    "KeyGuessAttempts", "SignatureSize", "SpoofCertMatch", "TimingVariance",
    "LeakageSignal",
]
CATEGORICAL_COLS = ["rsu_zone", "msg_type", "source", "vehicle_type", "protocol"]
H5_EXTRA_NUMERIC_COLS = [
    "ZoneRisk_tunnel", "cos_timestamp", "sin_timestamp", "decryption_time",
    "enc_dec_ratio", "latency_ber_ratio", "risk_amp", "snr_packet_loss_ratio",
]


def _json_safe(obj):
    # subclass_metrics's per_class dict has numpy.int64 keys (from the
    # class labels), which json.dumps rejects on this platform/Python
    # build. Recursively normalize numpy scalar keys/values to native
    # Python types before serializing.
    if isinstance(obj, dict):
        return {
            (k.item() if isinstance(k, np.generic) else k): _json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run_corpus(name, parquet_path, extra_numeric_cols, model_names=None):
    df = pd.read_parquet(parquet_path)
    # The raw attack_family column is NaN for every Normal row and for ~4%
    # of attack rows (confirmed against the real CSV during plan review) —
    # not a clean 4-class label. Overwrite it with the derived version,
    # which is built from class_label (100% populated) and always agrees
    # with the README's documented family distribution.
    df[FAMILY_LABEL_COL] = derive_family_label(df["class_label"])

    # class_label and attack_family(now derived) are labels, not features —
    # kept for splitting/training targets, even though they're in
    # LEAKY_COLUMNS as inputs to the *feature matrix* built inside
    # train_stage1/train_stage2.
    label_cols = {FAMILY_LABEL_COL, "class_label"}
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns and c not in label_cols])

    numeric_cols = NUMERIC_COLS + [c for c in extra_numeric_cols if c in df.columns]
    train_df, val_df, test_df = stratified_split(
        df, label_col="class_label", group_col=SPLIT_GROUP_COL_BY_CORPUS[name], seed=SEED
    )

    stage1 = train_stage1(
        train_df, val_df, numeric_cols=numeric_cols, categorical_cols=CATEGORICAL_COLS,
        model_names=model_names,
    )
    best_stage1_preds = stage1["models"]["lightgbm"]["val_predictions"]
    stage1_metrics = family_metrics(stage1["family_true"], best_stage1_preds)

    stage2 = train_stage2(
        train_df, val_df, numeric_cols=numeric_cols, categorical_cols=CATEGORICAL_COLS,
        model_names=model_names,
    )
    best_stage2_preds = stage2["models"]["lightgbm"]["val_predictions"]
    stage2_metrics = subclass_metrics(stage2["subclass_true"], best_stage2_preds)

    real_synth_df = val_df.iloc[: len(best_stage1_preds)].copy()
    real_synth_df["y_true"] = stage1["family_true"]
    real_synth_df["y_pred"] = best_stage1_preds
    real_vs_synth = compare_real_synthetic(real_synth_df, "y_true", "y_pred")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}_stage1_metrics.json").write_text(json.dumps(_json_safe(stage1_metrics), indent=2))
    (RESULTS_DIR / f"{name}_stage2_metrics.json").write_text(json.dumps(_json_safe(stage2_metrics), indent=2))
    (RESULTS_DIR / f"{name}_real_vs_synthetic.json").write_text(json.dumps(_json_safe(real_vs_synth), indent=2))

    return {"stage1": stage1_metrics, "stage2": stage2_metrics}


if __name__ == "__main__":
    csv_parquet = csv_to_parquet(RAW_CSV_PATH, CSV_CORPUS_DIR)
    h5_parquet = h5_to_parquet(RAW_H5_PATH, H5_CORPUS_DIR)

    csv_results = run_corpus("csv", csv_parquet, extra_numeric_cols=[], model_names=REAL_RUN_MODEL_NAMES)
    h5_results = run_corpus(
        "h5", h5_parquet, extra_numeric_cols=H5_EXTRA_NUMERIC_COLS, model_names=REAL_RUN_MODEL_NAMES
    )

    comparison = {
        "csv_stage1_macro_f1": csv_results["stage1"]["macro_f1"],
        "h5_stage1_macro_f1": h5_results["stage1"]["macro_f1"],
        "csv_stage2_macro_f1": csv_results["stage2"]["macro_f1"],
        "h5_stage2_macro_f1": h5_results["stage2"]["macro_f1"],
    }
    (RESULTS_DIR / "csv_vs_h5_comparison.json").write_text(json.dumps(_json_safe(comparison), indent=2))
    print(json.dumps(_json_safe(comparison), indent=2))
```

- [ ] **Step 2: Confirm `SPLIT_GROUP_COL_BY_CORPUS` and `REAL_RUN_MODEL_NAMES`**

`SPLIT_GROUP_COL_BY_CORPUS` is already set to `{"csv": "session_id", "h5": None}` above (CSV has real session grouping per Task 4; H5's session_id is a constant 0.0 placeholder in this release, confirmed by direct inspection, so it falls back to plain stratified splitting) and `REAL_RUN_MODEL_NAMES` is already set to `["lightgbm"]` (per the timing finding above). Just confirm `src/run_pipeline.py` has both values before running — do not silently change either back.

- [ ] **Step 3: Run the full pipeline against real data**

```bash
python -m src.run_pipeline
```

Expected runtime: LightGBM-only across 2 stages × 2 corpora on ~3.5M training rows should be well under an hour (LightGBM's fit time was a small fraction of the 40k-row smoke test's Stage 1 total, unlike RandomForest/MLP) — but this is still the slowest step in the plan; run it in the background and check on it periodically rather than blocking on a single call, and don't assume a bail-out threshold from the old 4-model estimate applies here.

- [ ] **Step 4: Inspect the results**

```bash
python -c "
import json
from pathlib import Path
for p in sorted(Path('results').glob('*.json')):
    print('---', p.name, '---')
    print(json.dumps(json.load(open(p)), indent=2)[:500])
"
```

Confirm: `csv_vs_h5_comparison.json` has four macro-F1 numbers (hypothesis 2's core evidence); each `*_real_vs_synthetic.json` has both a `"real"` and `"synthetic"` block (hypothesis 3's evidence); each `*_stage1_metrics.json`/`*_stage2_metrics.json` has a non-degenerate confusion matrix / per-class breakdown (hypothesis 1's evidence).

- [ ] **Step 5: Commit the code and the results artifacts**

```bash
git add src/run_pipeline.py results/*.json
git commit -m "feat: add end-to-end orchestration and real-data pipeline results"
```

---

## Task 14 Amendment 2 (2026-08-27, after the final whole-branch review)

The final review (most capable model, full 23-commit range) found two Critical,
data-integrity-affecting defects in the committed real-run results, independently
verified by the controller before acting on them:

**C1 — H5 train/val split leaks 99.98% duplicate rows.** Direct inspection of
the real CSV parquet confirms both dataset releases are **not** 5,000,000
independent observations: `pq_v2x_realistic.csv` has exactly 100,000 unique
combinations of every model-input feature column, each replicated exactly 50
times (`session_id` is a 1:1 proxy for this — grouping by it, as Task 14
already did, happens to be exactly correct for CSV). H5 has an analogous
~105,000-unique-row structure but no usable key exposing it (`session_id` is
the constant `0.0` placeholder found in Amendment 1), so its plain-stratified
fallback scattered each duplicate block's 50 copies across train/val/test —
99.9785% of H5 validation rows had an exact feature-twin in training. This
inflated H5's committed macro-F1 (0.980/0.994) — a leakage-free re-run of H5
Stage 1 measured 0.878, still good but not the near-perfect result on file.

**C2 — 15 sparse crypto-trace columns + `is_real_trace` never reached the
model.** `NUMERIC_COLS`/`CATEGORICAL_COLS` in `src/run_pipeline.py` never
included `SPARSE_TRACE_COLUMNS` or `is_real_trace`, so `add_missingness_indicators`
was never triggered (its `sparse_present` gate required the raw column to
already be in `numeric_cols + categorical_cols` — a circular condition that
was never true) and the real/synthetic distinction was invisible to the
model. `*_real_vs_synthetic.json`'s hypothesis-3 evidence is not meaningful
as committed.

The user reviewed these findings and decided: fix both, re-run the full
5,000,000-row pipeline. Two other Important findings from the same review
(no flat 24-class baseline for hypothesis 1; no Stage-1-routed cascaded
end-to-end metric) are **out of scope for this amendment** — documented as
known limitations, not implemented.

### Fix 1 — duplicate-aware split (replaces `SPLIT_GROUP_COL_BY_CORPUS`)

Both corpora's true duplicate structure is exactly captured by hashing the
row's own feature values — this is a direct generalization of what grouping
by CSV's `session_id` was already doing by coincidence (proven: CSV's
`session_id` and its 100,000 unique feature combinations are in exact 1:1
correspondence). Delete `SPLIT_GROUP_COL_BY_CORPUS` and its lookup in
`run_corpus`; replace with a per-row hash of the actual feature columns,
computed identically for both corpora:

```python
# in run_corpus, after numeric_cols is finalized and before stratified_split:
df["_dedup_group"] = pd.util.hash_pandas_object(
    df[numeric_cols + CATEGORICAL_COLS], index=False
)

train_df, val_df, test_df = stratified_split(
    df, label_col="class_label", group_col="_dedup_group", seed=SEED
)

# Safety net: confirm the property the whole grouped-split mechanism exists
# to guarantee. Cheap relative to training; keep it permanently, not just
# for this one run.
overlap = (
    set(train_df["_dedup_group"]) & set(val_df["_dedup_group"])
) | (
    set(train_df["_dedup_group"]) & set(test_df["_dedup_group"])
) | (
    set(val_df["_dedup_group"]) & set(test_df["_dedup_group"])
)
assert not overlap, (
    f"{name}: {len(overlap)} duplicate-feature groups leaked across splits"
)
```

`_dedup_group` must never be added to `numeric_cols`/`categorical_cols` (it
isn't a feature) — `ColumnTransformer`'s default `remainder="drop"` already
excludes it from the model input as long as it's absent from those two lists,
matching how every other non-feature column in `df` is already handled.

### Fix 2 — include sparse trace columns and `is_real_trace` as features

**`src/models_stage1.py` and `src/models_stage2.py`**: the `sparse_present`
line currently reads `[c for c in SPARSE_TRACE_COLUMNS if c in numeric_cols +
categorical_cols]` — a circular gate (a column only gets a missingness
indicator if it's already a feature, so a column that's *only* meant to
contribute its missingness — never its raw value — can never get one).
Change both occurrences (`models_stage1.py`, `models_stage2.py`) to:

```python
sparse_present = [c for c in SPARSE_TRACE_COLUMNS if c in train_df.columns]
```

This decouples "does this column get a `{col}_is_missing` indicator" (now:
whenever the raw column exists in the corpus at all) from "is the column's
raw value itself also used as a feature" (still controlled by whether it's
in the `numeric_cols`/`categorical_cols` arguments) — needed because three
of the fifteen `SPARSE_TRACE_COLUMNS` (see below) get an indicator but must
never have their raw value fed to the model.

**`src/run_pipeline.py`**: verified real dtypes against the actual corpus
before writing this list — `protocol_mismatch`/`zone_mismatch_flag` are
`object` (True/False/NaN, not a clean numeric dtype — categorical is
correct), `cipher_suite`/`block_mode` are low-cardinality strings (2-3
values), the other numeric sparse fields are `float64`, `is_real_trace` is
`bool` with zero nulls. `pqc_public_key`/`pqc_ciphertext`/`pqc_signature`
are raw cryptographic blob strings — almost-unique per non-null row, so
one-hot-encoding their *value* would explode `OneHotEncoder`'s output
dimensionality for no modeling benefit; only their *presence/absence*
(already handled by Fix 2's indicator change above) is meaningful. Extend
the constants:

```python
NUMERIC_COLS = [
    "speed", "acceleration", "direction", "snr", "packet_loss", "latency",
    "ber", "KeyReuseCount", "EncapTimeDeviation", "EntropyDeviation",
    "KeyGuessAttempts", "SignatureSize", "SpoofCertMatch", "TimingVariance",
    "LeakageSignal",
    "sequence_gap", "timestamp_offset", "message_frequency", "source_entropy",
    "handshake_time", "encryption_time", "sig_generation_time",
    "sig_verification_time", "is_real_trace",
]
CATEGORICAL_COLS = [
    "rsu_zone", "msg_type", "source", "vehicle_type", "protocol",
    "protocol_mismatch", "zone_mismatch_flag", "cipher_suite", "block_mode",
]
```

`pqc_public_key`, `pqc_ciphertext`, `pqc_signature` are deliberately **not**
added to either list — their `_is_missing` indicators (from Fix 2's
`models_stage1.py`/`models_stage2.py` change) are the only trace of them
that reaches the model, which is the correct behavior for a field whose raw
value is not learnable signal but whose presence is.

### Re-verification before re-running at full scale

1. Re-run `tests/test_models_stage1.py`, `tests/test_models_stage2.py` (the
   `sparse_present` change touches both) — must still pass; the existing
   tests use `numeric_cols`/`categorical_cols` that don't include any
   `SPARSE_TRACE_COLUMNS` name, so `sparse_present` should still evaluate to
   `[]` for them (no `SPARSE_TRACE_COLUMNS` entry is a synthetic test
   fixture column) and behavior should be unchanged.
2. Re-run the existing smoke test (`scripts/smoke_test_pipeline.py`) against
   both corpora and confirm the new `assert not overlap` passes (proving
   the split fix works) before committing to the ~3-4 hour full-scale
   re-run.
3. Re-run `python -m src.run_pipeline` against the real 5,000,000-row data,
   overwriting `results/*.json`.
4. Commit the four modified files (`src/models_stage1.py`,
   `src/models_stage2.py`, `src/run_pipeline.py`, plus updated
   `results/*.json`) together.

---

## Self-Review Notes

- **Spec coverage:** Corpus identity (Task 2, 4), session leakage (Task 3, 4), H5 leakage audit (Task 7, 8), leakage-safe feature policy (Task 7, 9), Parquet conversion (Task 5), stratified/grouped split (Task 6), 4-model comparison at both stages (Task 10, 11, 12), all five evaluation angles — family, subclass, end-to-end, CSV-vs-H5, real-vs-synthetic (Task 13, 14) — are all covered.
- **Placeholder scan:** no TBD/TODO; every step has runnable code.
- **Type consistency:** `train_and_eval_models` return shape (`{"model", "val_predictions"}`) is used identically in Task 11, 12, and 14; `FAMILY_LABEL_COL` is defined once in `models_stage1.py` and imported everywhere else it's needed (Task 12, 14) rather than redefined.
- **Pre-flight correctness fix (added during SDD pre-flight scan, before Task 1 dispatch):** the raw `attack_family` column is NaN for every Normal row and ~4% of attack rows (verified against the real CSV) — Task 14 originally read it directly as the Stage 1 label, which would have trained Stage 1 on a broken/partial label. Fixed by adding `CLASS_LABEL_TO_FAMILY`/`derive_family_label` to Task 7 (derived from the always-populated `class_label`) and having Task 14 overwrite `df[FAMILY_LABEL_COL]` with the derived value before splitting/training. class_label and label_cols kept out of the leaky-column drop in Task 14 accordingly.
