# PQ-V2X Hierarchical PQC Threat Detection — Experimental Design

Status: approved (hypothesis-testing scope only; paper writing is out of scope for this spec)
Date: 2026-08-26

## Goal

Test, experimentally, whether a hierarchical (family → subclass) ML pipeline
can detect post-quantum (PQ) and hybrid cryptographic-transition attacks in
the PQ-V2X dataset, and whether the extended 65-column release (H5) adds
real detection power over the original 55-column paper-era release (CSV)
for this specific task. This is a new, independent study built on the
PQ-V2X dataset — not a reproduction of the original PQ-V2X paper benchmark.

Paper writing (structure, narrative, venue formatting) is explicitly out of
scope for this spec. This covers only the data/modeling/evaluation pipeline
needed to test the hypotheses below.

## Hypotheses under test

1. A two-stage hierarchical classifier (Stage 1: Normal / Post-Quantum /
   Hybrid / Classical family; Stage 2: specific subclass within PQ+Hybrid)
   detects PQC-related attacks at least as well as a flat 24-class
   classifier, while giving more interpretable per-family error analysis.
2. The H5 extended release's derived features (`enc_dec_ratio`,
   `sign_verify_ratio`, `decryption_time`, `ZoneRisk_tunnel`,
   `sin_timestamp`/`cos_timestamp`, `risk_amp`, `size_anomaly`,
   `snr_packet_loss_ratio`, `latency_ber_ratio`) measurably improve
   detection of PQ/Hybrid subclasses versus the CSV's 55 columns alone.
3. Detection quality differs meaningfully between `is_real_trace=True`
   (real cryptographic execution traces, ~10% of rows) and
   `is_real_trace=False` (synthetic traces, ~90% of rows) — relevant to
   how much the results generalize beyond synthetic generation.

## Confirmed facts about the data (verified this session)

- `pq_v2x_realistic.csv`: 5,000,000 rows × 55 columns, matches the README
  exactly (class distribution in the first 100k rows already matches the
  documented global distribution — the file is pre-shuffled, not sorted by
  class).
- `pq_v2x_dataset.h5`: 5,000,000 rows × **65** fields (via h5py inspection).
  This does not match the CSV's 55-column schema — it is very likely the
  "later extended PQ-V2X release" the README explicitly warns not to
  silently substitute for the original 55-column benchmark. Column-count
  confirmed; identity of underlying rows (same corpus, extended, vs. a
  separately generated 5M-row corpus) is **not yet confirmed** — see Open
  Technical Questions below.
- Leakage audit (empirical, on the CSV, n=500k sample):
  - `risk_score`: mean 0.158 (normal) vs 0.425 (attack), essentially
    separable — must be excluded from model inputs (not currently listed
    in the README's exclusion list, but behaves as a post-hoc severity
    score).
  - `synthetic_anomaly_score`: mean 0.265 (normal) vs 0.537 (attack), same
    issue — must also be excluded.
  - `is_real_trace`: 39.9% attack rate when True vs 39.4% when False — not
    leaky, safe to use as an input feature.
  - README's already-documented exclusion list (`label`, `class_label`,
    `attack_type`, `is_attack`, `attack_family`, `attack_severity`,
    `detected_flag`, `detection_method`) is confirmed necessary and
    insufficient on its own — `risk_score` and `synthetic_anomaly_score`
    must be added to it.
- Sparse crypto-trace-detail columns (`pqc_public_key`, `handshake_time`,
  `cipher_suite`, `protocol_mismatch`, etc.) are populated in only ~1-6% of
  rows overall, and only ever populated when `is_real_trace=True` (e.g.
  `pqc_public_key` notna rate: 0% when False, ~60% when True). These need
  an explicit missing-data policy (see Feature Policy below), not naive
  imputation with global means.
- The H5 extended fields likely need the same leakage audit as the CSV
  fields before use — not yet performed. This is a first implementation
  step, not assumed safe by default.

## Open technical questions (must be resolved before modeling starts)

These are implementation-plan tasks, not open design decisions — they were
called out during brainstorming as things to verify with code, not to
decide by discussion:

1. **CSV/H5 corpus identity** (Resolved 2026-08-26): 
   positional_match_rate=0.0, set_overlap_rate=0.0, same_corpus_reordered=false
   → CSV and H5 are **not** the same underlying corpus. They are separate
   5M-row datasets, not paired observations. CSV-vs-H5 comparison (hypothesis 2)
   must be treated as two independent samples, not a paired comparison.
2. **Session-level leakage** (Resolved 2026-08-26):
   max_rows_per_group=5, leakage_risk=true (first 500k CSV rows)
   → session_id repeats: yes, up to 5 rows per session_id on average across 100k unique sessions
   in the first 500k rows. Split must be grouped by session_id to prevent
   train/test leakage, not row-random.
3. **H5 extended-field leakage audit** (Resolved 2026-08-26):
   Audited 9 H5-only derived fields (n=300,000 rows) against is_attack target:

   ```
   column  auc_vs_target  likely_leakage
                risk_amp       0.993206            True
         ZoneRisk_tunnel       0.811971           False
   snr_packet_loss_ratio       0.550209           False
       latency_ber_ratio       0.507629           False
           cos_timestamp       0.501981           False
           sin_timestamp       0.501981           False
            size_anomaly       0.501247           False
         decryption_time       0.500751           False
           enc_dec_ratio       0.500750           False
   ```

   Result: `risk_amp` flagged as likely_leakage (AUC=0.9932 >> threshold 0.97).
   Added to `LEAKY_COLUMNS` list. All other 8 fields are safe to use as model inputs.

## Pipeline architecture

```
[CSV 55-col] ──┐
               ├─→ [Clean + leakage-safe feature policy] ─→ [Stratified split] ─→ [Stage 1: family] ─→ [Stage 2: PQ/Hybrid subclass] ─→ [Evaluation]
[H5 65-col]  ──┘
```

Two parallel, structurally identical pipelines (CSV-only features vs.
H5-only features) feed the same split methodology and the same model
architecture, so results are comparable pairwise per hypothesis 2.

## Data strategy

- Convert both CSV and H5 to **Parquet** (partitioned) early — smaller,
  faster to load repeatedly, properly typed (avoids the CSV's mixed
  `object`/`NaN` columns). Kept in `data/processed/`, not versioned in git
  (large; original raw files stay in `data/raw/`, also not versioned).
- **Split**: stratified by `class_label` (all 24 classes, not just family)
  into train/val/test (70/15/15), fixed seed — ensures small attack
  classes (~1-3% each) are represented in every split. Upgraded to
  group-by-session if Open Technical Question 2 confirms session repeats.
- Stage 2 training data: PQ+Hybrid rows only (19 classes), using
  ground-truth family labels for the training set (not Stage-1 predicted
  routing) so Stage 2 training isn't contaminated by Stage-1 errors. Stage
  1's predicted routing is used only at evaluation time, to measure
  realistic end-to-end pipeline error.

## Feature policy

Excluded from all model inputs (confirmed leaky or target-derived):
`label`, `class_label`, `attack_type`, `is_attack`, `attack_family`,
`attack_severity`, `detected_flag`, `detection_method`, `risk_score`,
`synthetic_anomaly_score` (+ H5-only equivalents once audited per Open
Technical Question 3).

Sparse crypto-trace-detail columns (only populated when
`is_real_trace=True`): kept as features with explicit missingness
indicators (e.g. a `*_is_missing` boolean per sparse column) rather than
mean/mode imputation, since missingness itself is informative
(real-vs-synthetic trace path) — exact encoding is an implementation
detail for `src/features.py`.

## Modeling (CPU-only — no GPU available)

**Stage 1 (family: Normal/PQ/Hybrid/Classical)** and **Stage 2 (PQ+Hybrid
subclass, 19 classes)** both use the same four-model comparison:

- **LightGBM** — primary model (fast on CPU at 5M rows, native categorical
  support).
- **Logistic Regression** — linear baseline.
- **Random Forest** — classical non-linear baseline (common in IDS
  literature).
- **MLP** (`scikit-learn MLPClassifier`, 1-2 hidden layers) — lightweight
  deep-learning-style baseline.

No sequence models (LSTM/Transformer) — not justified on CPU-only hardware
for a tabular (non-sequential-per-row) problem; noted as a limitation, not
attempted.

**Class imbalance**: handled via LightGBM's native class weighting
(`class_weight`/`scale_pos_weight`-equivalent), not oversampling — SMOTE at
5M rows with mixed categorical/numeric features risks synthetic-artifact
distortion and is computationally heavy for CPU-only hardware.

## Evaluation

- Stage 1: accuracy, macro-F1, 4×4 confusion matrix across families.
- Stage 2: per-class precision/recall/macro-F1 across the 19 PQ/Hybrid
  subclasses, confusion analysis between specific attack pairs.
- End-to-end: pipeline metric using Stage-1-predicted routing feeding
  Stage 2 (measures realistic compounded error).
- **CSV vs. H5** (hypothesis 2): same metrics, same split, side by side per
  class.
- **Real vs. synthetic** (hypothesis 3): same metrics computed separately
  on `is_real_trace=True` vs `False` subsets.

## Repository structure

```
quantumTest/
├── data/
│   ├── raw/              # CSV and H5 originals (gitignored)
│   └── processed/        # Parquet (csv_corpus/, h5_corpus/), gitignored
├── src/
│   ├── data_prep.py      # Parquet conversion, corpus-identity check, session-leakage check, split
│   ├── features.py       # leakage-safe feature policy, missingness encoding, categorical encoding
│   ├── models_stage1.py  # family: LightGBM + LogReg + RF + MLP
│   ├── models_stage2.py  # PQ/Hybrid subclass: same 4 models
│   └── evaluate.py       # metrics, confusion matrices, CSV-vs-H5, real-vs-synthetic comparisons
├── notebooks/             # ad hoc exploration (optional)
├── results/                # generated metrics/figures, versioned in git as reproducibility artifacts
├── requirements.txt        # pandas, pyarrow, lightgbm, scikit-learn, h5py
└── docs/superpowers/specs/ # this design doc
```

Reproducibility: single fixed seed used everywhere; hyperparameters and
metrics logged as JSON/CSV under `results/` (no MLflow needed at this
scope). This becomes a git repo (currently isn't one) to version code and
results; raw/processed data stays out of git via `.gitignore`.

## Explicitly out of scope for this spec

- Paper structure, narrative, venue formatting — deferred entirely per
  user direction ("não se preocupe com o paper agora, apenas em testar as
  hipóteses").
- Deep sequence models (LSTM/Transformer).
- Cloud/GPU training (CPU-only per user's stated resources).
