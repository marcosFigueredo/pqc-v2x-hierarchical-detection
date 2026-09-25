"""Controlled ablation for the leakage claim (IEEE OJ-ITS review, M3).

The manuscript contrasts the original PQ-V2X benchmark's near-perfect
macro-F1 against this study's much lower values, and then concedes that the
two numbers "do not measure exactly the same task" — different sample sizes,
different split procedure, different scale. That concession is the weakest
point in the paper's most provocative claim, and the dataset's own authors
are among its likely reviewers.

This script removes the concession. It runs the identical model on the
identical data at the identical scale, changing exactly one thing: whether
the split is grouped by feature identity or is a plain stratified random
split of the kind the original benchmark used. Whatever the difference turns
out to be, it is then attributable to the split and to nothing else.

LightGBM only, both releases, primary seed — the point is the contrast, not
a model sweep.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.run_pipeline as pipeline
from src.config import CSV_CORPUS_DIR, H5_CORPUS_DIR, RESULTS_DIR, SEED

MODEL_NAMES = ["lightgbm"]
CORPORA = (
    ("csv", CSV_CORPUS_DIR / "data.parquet", []),
    ("h5", H5_CORPUS_DIR / "data.parquet", pipeline.H5_EXTRA_NUMERIC_COLS),
)


def main():
    out_dir = RESULTS_DIR / "split_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    original_results_dir = pipeline.RESULTS_DIR

    summary = {}
    try:
        for grouped in (True, False):
            condition = "grouped" if grouped else "random"
            for name, path, extra in CORPORA:
                run_dir = out_dir / condition / name
                run_dir.mkdir(parents=True, exist_ok=True)
                pipeline.RESULTS_DIR = run_dir
                print(f"=== split ablation: {name} / {condition} ===", flush=True)

                t0 = time.time()
                result = pipeline.run_corpus(
                    name, path, extra, model_names=MODEL_NAMES, seed=SEED,
                    bootstrap=False, grouped_split=grouped,
                )
                summary.setdefault(condition, {})[name] = {
                    "stage1_macro_f1": result["stage1"]["macro_f1"],
                    "stage1_accuracy": result["stage1"]["accuracy"],
                    "flat_24class_macro_f1": result["flat"]["flat_24class"]["macro_f1"],
                    "stage2_oracle_macro_f1": result["stage2"]["macro_f1"],
                    "test_rows_with_feature_twin_in_train":
                        result["provenance"]["test_rows_with_feature_twin_in_train"],
                    "wall_clock_seconds": round(time.time() - t0, 1),
                }
    finally:
        pipeline.RESULTS_DIR = original_results_dir

    # The headline of the ablation: how much macro-F1 the split alone buys.
    for name, _, _ in CORPORA:
        grouped = summary["grouped"][name]
        random_ = summary["random"][name]
        summary.setdefault("inflation", {})[name] = {
            "stage1_macro_f1_delta": random_["stage1_macro_f1"] - grouped["stage1_macro_f1"],
            "flat_24class_macro_f1_delta":
                random_["flat_24class_macro_f1"] - grouped["flat_24class_macro_f1"],
            "stage2_oracle_macro_f1_delta":
                random_["stage2_oracle_macro_f1"] - grouped["stage2_oracle_macro_f1"],
        }

    output = out_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
