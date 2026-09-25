"""Throwaway smoke test for Task 14's run_pipeline logic.

Runs the same code path as src/run_pipeline.py but against small samples of
the real CSV/H5 files (first N rows) written to a separate smoke-test data
dir, to catch dtype/NaN/categorical surprises fast before committing to the
full 5M-row run. Not part of the deliverable.
"""
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RAW_CSV_PATH, RAW_H5_PATH, RESULTS_DIR, SEED
from src.data_prep import csv_to_parquet, h5_to_parquet
from src.run_pipeline import H5_EXTRA_NUMERIC_COLS, REAL_RUN_MODEL_NAMES, run_corpus

N = 40_000

SMOKE_DIR = PROJECT_ROOT / "data" / "smoke"
SMOKE_CSV = SMOKE_DIR / "sample.csv"
SMOKE_H5 = SMOKE_DIR / "sample.h5"
SMOKE_CSV_CORPUS = SMOKE_DIR / "csv_corpus"
SMOKE_H5_CORPUS = SMOKE_DIR / "h5_corpus"
SMOKE_RESULTS = SMOKE_DIR / "results"


def make_samples():
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sampling first {N} rows of CSV -> {SMOKE_CSV}")
    df = pd.read_csv(RAW_CSV_PATH, nrows=N, low_memory=False)
    df.to_csv(SMOKE_CSV, index=False)

    print(f"Sampling first {N} rows of H5 -> {SMOKE_H5}")
    with h5py.File(RAW_H5_PATH, "r") as fin, h5py.File(SMOKE_H5, "w") as fout:
        for col in fin.keys():
            values = fin[col][:N]
            fout.create_dataset(col, data=values)


def make_samples_from_parquet():
    """Build smoke Parquets from the local cache when Drive is unreadable.

    The processed cache is the exact input consumed by ``run_corpus``.  This
    fallback keeps the smoke test runnable on machines where the raw files
    are mounted but temporarily unavailable to the Python process.
    """
    from src.config import CSV_CORPUS_DIR, H5_CORPUS_DIR

    SMOKE_CSV_CORPUS.mkdir(parents=True, exist_ok=True)
    SMOKE_H5_CORPUS.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (CSV_CORPUS_DIR / "data.parquet", SMOKE_CSV_CORPUS / "data.parquet"),
        (H5_CORPUS_DIR / "data.parquet", SMOKE_H5_CORPUS / "data.parquet"),
    ):
        print(f"Sampling first {N} rows of cached Parquet -> {target}")
        pd.read_parquet(source).head(N).to_parquet(target, index=False)


def main():
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    try:
        make_samples()
        print("Converting smoke CSV -> Parquet")
        csv_parquet = csv_to_parquet(SMOKE_CSV, SMOKE_CSV_CORPUS, chunksize=10_000)
        print("Converting smoke H5 -> Parquet")
        h5_parquet = h5_to_parquet(SMOKE_H5, SMOKE_H5_CORPUS, chunksize=10_000)
    except (PermissionError, OSError) as exc:
        print(f"Raw source unavailable ({exc}); falling back to cached Parquet")
        make_samples_from_parquet()
        csv_parquet = SMOKE_CSV_CORPUS / "data.parquet"
        h5_parquet = SMOKE_H5_CORPUS / "data.parquet"

    # Monkeypatch RESULTS_DIR so the smoke test doesn't clobber real results.
    import src.run_pipeline as rp
    rp.RESULTS_DIR = SMOKE_RESULTS

    t0 = time.time()
    print(f"=== Smoke: CSV corpus (model_names={REAL_RUN_MODEL_NAMES}) ===")
    csv_results = run_corpus("csv", csv_parquet, extra_numeric_cols=[], model_names=REAL_RUN_MODEL_NAMES)
    print(f"=== Smoke: H5 corpus (model_names={REAL_RUN_MODEL_NAMES}) ===")
    h5_results = run_corpus(
        "h5", h5_parquet, extra_numeric_cols=H5_EXTRA_NUMERIC_COLS, model_names=REAL_RUN_MODEL_NAMES
    )
    print(f"Smoke test finished in {time.time() - t0:.1f}s")
    print("CSV stage1 macro_f1:", csv_results["stage1"]["macro_f1"])
    print("CSV stage2 macro_f1:", csv_results["stage2"]["macro_f1"])
    print("H5 stage1 macro_f1:", h5_results["stage1"]["macro_f1"])
    print("H5 stage2 macro_f1:", h5_results["stage2"]["macro_f1"])
    print("CSV flat 24class macro_f1:", csv_results["flat"]["flat_24class"]["macro_f1"])
    print("CSV flat family-derived macro_f1:", csv_results["flat"]["flat_family_derived"]["macro_f1"])
    print("CSV flat subtype(PQ/Hybrid) macro_f1:", csv_results["flat"]["flat_subtype_pq_hybrid"]["macro_f1"])
    print("H5 flat 24class macro_f1:", h5_results["flat"]["flat_24class"]["macro_f1"])
    print("H5 flat family-derived macro_f1:", h5_results["flat"]["flat_family_derived"]["macro_f1"])
    print("H5 flat subtype(PQ/Hybrid) macro_f1:", h5_results["flat"]["flat_subtype_pq_hybrid"]["macro_f1"])
    print("CSV stage2 REAL-routing macro_f1:", csv_results["real_cascade"]["stage2_real_routing"]["macro_f1"])
    print("CSV flat REAL-routing macro_f1:", csv_results["real_cascade"]["flat_real_routing"]["macro_f1"])
    print("H5 stage2 REAL-routing macro_f1:", h5_results["real_cascade"]["stage2_real_routing"]["macro_f1"])
    print("H5 flat REAL-routing macro_f1:", h5_results["real_cascade"]["flat_real_routing"]["macro_f1"])


if __name__ == "__main__":
    main()
