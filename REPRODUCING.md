# Reproducing the results

This is the complete computational record for *Reversal of Hierarchical
Advantage in V2X Intrusion Detection for Post-Quantum Intelligent
Transportation Systems*. Everything reported in the manuscript is produced by
the code in this archive from the public PQ-V2X dataset.

## What is and is not included

Included: all source code, the evaluation protocol, hyperparameters, seeds,
the automated test suite, the run logs, and every results file the manuscript's
tables are built from (`results/*.json`).

Not included: the PQ-V2X data itself (about 8.7 GB across the two releases).
It is distributed by its own authors; see the dataset paper
(doi:10.1109/JIOT.2025.3618153). Place the two raw files where `src/config.py`
expects them, or edit that file's paths.

## Environment

Python 3.11.0. The results in the manuscript were produced on Windows with an
Intel Core i5-1135G7 and 32 GB of RAM, entirely on CPU, with no GPU used.

```
pip install -r requirements.txt
```

Pinned versions that produced the published numbers: pandas 3.0.3,
pyarrow 25.0.0, h5py 3.16.0, scikit-learn 1.6.1, lightgbm 4.6.0, numpy 1.26.4.

## Running

```
python -m pytest tests/ -q                       # protocol checks, seconds
python -m src.run_pipeline                       # main run, four models, hours
python scripts/run_multiseed_robustness.py       # seeds 42/123/2024, LightGBM
python scripts/run_random_split_ablation.py      # grouped vs random split
```

The first converts the raw releases to Parquet on first use and reuses that
cache afterwards.

## The evaluation protocol, in one page

Two PQ-V2X releases, 5,000,000 records each, 24 subtype labels grouped into
four families (Normal, Classical, Hybrid, PQ).

**Leakage control.** Columns are audited against `is_attack` by AUC and those
at or above 0.97 are dropped as leaks. Neither release contains 5,000,000
independent observations: the original holds exactly 100,000 unique predictor
combinations repeated 50 times each, the extended about 105,000. Splitting at
random therefore places exact copies of the same pattern on both sides of the
train/test boundary. Every split here is stratified by label and grouped by a
hash of the row's own predictor values, and `run_corpus` asserts on every run
that no feature group spans partitions.

**Which partition is scored.** Training uses the 70% train split. Every
reported metric comes from the 15% *test* split. The 15% validation split is
deliberately never read — nothing is tuned or early-stopped on it. Each
results file records this in its `provenance` block, and
`tests/test_run_pipeline_partition.py` fails if the evaluation partition ever
moves.

**What is compared.** A flat 24-class classifier against a two-stage cascade
(Stage 1 predicts family, Stage 2 predicts subtype over the 19 PQ/Hybrid
labels). Stage 2 is scored three ways, and the three answer different
questions:

| Condition | Rows scored | Charges the cascade for |
|---|---|---|
| Oracle | true family is PQ/Hybrid | nothing from routing |
| Predicted routing | Stage 1 routed the row | routing false positives |
| End-to-end | true family is PQ/Hybrid | both false positives and false negatives |

Under the end-to-end condition a truly PQ/Hybrid row that Stage 1 never routed
receives a sentinel prediction that cannot be correct, so missed attacks cost
recall instead of silently leaving the evaluation set. The flat classifier has
no routing stage and never abstains, so it is scored over the identical rows.

**Floors and intervals.** Every task reports majority-class and
stratified-random macro-F1 so the reported values can be read against a floor.
The headline flat-versus-cascade difference carries a paired bootstrap
confidence interval over 1,000 resamples of the same rows.

## Reading the results files

`results/{csv,h5}_stage1_metrics.json`, `_stage2_metrics.json`,
`_flat_metrics.json`, `_real_cascade_metrics.json`;
`results/multiseed/summary.json`; `results/split_ablation/summary.json`.
`csv` is the original release, `h5` the extended one — these name the two
releases, not a serialization choice.
