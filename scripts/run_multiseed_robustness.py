"""Run the primary LightGBM experiment across multiple random seeds.

This is a robustness analysis for the paper, separate from the expensive
four-model full run.  Each seed gets its own directory, so the experiment is
restartable and never overwrites the main results.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.run_pipeline as pipeline
from src.config import CSV_CORPUS_DIR, H5_CORPUS_DIR, RESULTS_DIR


SEEDS = (42, 123, 2024)
MODEL_NAMES = ["lightgbm"]


def main():
    all_results = {}
    original_results_dir = pipeline.RESULTS_DIR

    try:
        for seed in SEEDS:
            seed_dir = RESULTS_DIR / "multiseed" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            pipeline.RESULTS_DIR = seed_dir
            print(f"=== Robustness run seed={seed} ===", flush=True)

            # The paired bootstrap is run once, on the primary seed, in the
            # main pipeline run; repeating 1,000 resamples per seed would
            # cost hours to re-measure the same quantity. Across-seed spread
            # is what this script is for.
            csv_result = pipeline.run_corpus(
                "csv", CSV_CORPUS_DIR / "data.parquet", [],
                model_names=MODEL_NAMES, seed=seed, bootstrap=False,
            )
            h5_result = pipeline.run_corpus(
                "h5", H5_CORPUS_DIR / "data.parquet", pipeline.H5_EXTRA_NUMERIC_COLS,
                model_names=MODEL_NAMES, seed=seed, bootstrap=False,
            )
            all_results[str(seed)] = {"csv": csv_result, "h5": h5_result}
    finally:
        pipeline.RESULTS_DIR = original_results_dir

    # Keep a compact summary for tables; the per-seed run directories retain
    # the complete confusion matrices, per-class metrics, and diagnostics.
    summary = {}
    for seed, corpora in all_results.items():
        summary[seed] = {}
        for corpus, result in corpora.items():
            cascade = result["real_cascade"]
            summary[seed][corpus] = {
                "stage1_macro_f1": result["stage1"]["macro_f1"],
                "stage2_oracle_macro_f1": result["stage2"]["macro_f1"],
                "stage2_real_routing_macro_f1": cascade["stage2_real_routing"]["macro_f1"],
                "flat_24class_macro_f1": result["flat"]["flat_24class"]["macro_f1"],
                "flat_real_routing_macro_f1": cascade["flat_real_routing"]["macro_f1"],
                # The end-to-end pair scores every truly PQ/Hybrid row, so
                # rows Stage 1 never routed cost the cascade recall instead
                # of leaving the evaluation set. Tracked per seed because the
                # two conditions can order the architectures differently.
                "stage2_end_to_end_macro_f1": cascade["stage2_end_to_end"]["macro_f1"],
                "flat_end_to_end_macro_f1": cascade["flat_end_to_end"]["macro_f1"],
                "routing_precision": cascade["routing_diagnostics"]["routing_precision"],
                "routing_recall": cascade["routing_diagnostics"]["routing_recall"],
                "evaluation_partition": result["provenance"]["evaluation_partition"],
            }

    output = RESULTS_DIR / "multiseed" / "summary.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
