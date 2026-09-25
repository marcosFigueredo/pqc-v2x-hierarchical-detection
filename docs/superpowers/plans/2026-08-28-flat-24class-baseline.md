# Flat 24-Class Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a single flat 24-class classifier (no family/subtype decomposition) on the same real data, split, and features already used by Stage 1/Stage 2, so Hypothesis 1 ("hierarchical beats flat") has an actual flat baseline to compare against.

**Architecture:** Add a third pipeline stage, `src/models_flat.py::train_flat`, that mirrors `train_stage1`'s shape exactly but trains directly on the full 24-class `class_label` over ALL rows (no Post-Quantum/Hybrid filtering). Wire it into `run_pipeline.py::run_corpus` alongside the existing Stage 1/Stage 2 calls, reusing the already-generic `subclass_metrics`/`family_metrics` from `src/evaluate.py` for three comparisons: (a) the flat model's raw 24-class macro-F1 (headline baseline number), (b) the flat model's predictions collapsed to family via `derive_family_label`, evaluated against the same rows Stage 1 sees — directly comparable to `stage1_metrics.macro_f1`, and (c) the flat model's raw subtype predictions restricted to the same oracle-family-filtered subset Stage 2 sees — directly comparable to `stage2_metrics.macro_f1`. This gets a fair, apples-to-apples comparison without needing the real Stage-1-routed cascade (a separate, larger gap tracked separately in STATUS.md item 2 — out of scope here).

**Tech Stack:** Python, pandas, scikit-learn (ColumnTransformer/Pipeline), LightGBM — same stack as the rest of `src/`.

**Spec:** [STATUS.md](../../../STATUS.md) item 1 under "O que falta decidir" (the flat-baseline gap this plan closes) and [PLANO_DE_ESTUDO.md](../../../PLANO_DE_ESTUDO.md) lines 1734-1738 (Task 14 Amendment 2 — flat baseline explicitly deferred as out of scope there, now being picked up).

## Global Constraints

- Reuse `SEED = 42` from `src/config.py` — never hardcode a different seed.
- Reuse the existing `_dedup_group`-based split already computed once per corpus in `run_corpus` — do not re-split; the flat baseline must train/validate on the exact same `train_df`/`val_df` rows Stage 1 sees, so the comparison is apples-to-apples.
- Reuse `NUMERIC_COLS`/`CATEGORICAL_COLS`/`extra_numeric_cols` and the `SPARSE_TRACE_COLUMNS` missingness-indicator pattern exactly as `train_stage1`/`train_stage2` already do — do not invent a new feature set.
- Real run trains **LightGBM only** (`REAL_RUN_MODEL_NAMES = ["lightgbm"]`), per the existing Task 14 Amendment 1 timing decision — the flat baseline follows the same `model_names` parameter, not a hardcoded model list.
- Follow the existing one-module-per-stage convention (`models_stage1.py`, `models_stage2.py` → `models_flat.py`), each with its own test file — do not fold this into an existing module.

---

## Task 1: `train_flat` in `src/models_flat.py`

**Files:**
- Create: `src/models_flat.py`
- Test: `tests/test_models_flat.py`

**Interfaces:**
- Consumes: `src.features.SPARSE_TRACE_COLUMNS`, `add_missingness_indicators`, `build_preprocessor` (existing); `src.model_training.train_and_eval_models` (existing).
- Produces: `train_flat(train_df, val_df, numeric_cols, categorical_cols, label_col="class_label", model_names=None) -> {"class_true": np.ndarray, "models": {name: {"model": ..., "val_predictions": np.ndarray}}}` — same shape as `train_stage1`'s return, but keyed `"class_true"` (not `"family_true"`) since the values are 24-class labels, not family labels. No row filtering is applied (unlike `train_stage2`, which filters to PQ/Hybrid).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_flat.py
import numpy as np
import pandas as pd

from src.models_flat import train_flat


def _synthetic_df(n):
    rng = np.random.default_rng(2)
    class_label = rng.integers(0, 4, size=n)
    return pd.DataFrame(
        {
            "class_label": class_label,
            "num_feat": class_label + rng.normal(0, 0.5, size=n),
            "cat_feat": rng.choice(["urban", "rural"], size=n),
        }
    )


def test_train_flat_returns_predictions_for_every_val_row_unfiltered():
    train_df = _synthetic_df(300)
    val_df = _synthetic_df(100)

    result = train_flat(
        train_df, val_df, numeric_cols=["num_feat"], categorical_cols=["cat_feat"],
    )

    assert len(result["class_true"]) == len(val_df)
    np.testing.assert_array_equal(result["class_true"], val_df["class_label"].values)
    for name, res in result["models"].items():
        assert res["val_predictions"].shape[0] == len(val_df)


def test_train_flat_uses_class_label_by_default():
    train_df = _synthetic_df(200)
    val_df = _synthetic_df(80)

    result = train_flat(train_df, val_df, numeric_cols=["num_feat"], categorical_cols=["cat_feat"])

    assert set(np.unique(result["class_true"])) <= set(val_df["class_label"].unique())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_models_flat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models_flat'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/models_flat.py
from src.features import SPARSE_TRACE_COLUMNS, add_missingness_indicators, build_preprocessor
from src.model_training import train_and_eval_models


def train_flat(train_df, val_df, numeric_cols, categorical_cols, label_col="class_label", model_names=None):
    sparse_present = [c for c in SPARSE_TRACE_COLUMNS if c in train_df.columns]

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

    models = train_and_eval_models(X_train, y_train, X_val, y_val, model_names=model_names)

    return {"class_true": y_val, "models": models}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_models_flat.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full existing suite to confirm no regressions**

Run: `python -m pytest -v`
Expected: all previously-passing tests (21 before this task) plus the 2 new ones pass; 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/models_flat.py tests/test_models_flat.py
git commit -m "feat: add flat 24-class baseline model training (no family/subtype split)"
```

---

## Task 2: Wire the flat baseline into `run_pipeline.py`

**Files:**
- Modify: `src/run_pipeline.py`
- Modify: `scripts/smoke_test_pipeline.py`

**Interfaces:**
- Consumes: `train_flat` from Task 1 (`src.models_flat`); existing `subclass_metrics`/`family_metrics` from `src.evaluate`; existing `derive_family_label` from `src.features` (already imported in `run_pipeline.py`); existing `FAMILY_LABEL_COL` from `src.models_stage1` (already imported).
- Produces: `run_corpus(...)` now returns `{"stage1": ..., "stage2": ..., "flat": {"flat_24class": ..., "flat_family_derived": ..., "flat_subtype_pq_hybrid": ...}}`; writes a new `results/{name}_flat_metrics.json` per corpus; the `__main__` block's `comparison` dict gains six new keys.

This task has no new automated test — `run_pipeline.py` has none today (its `run_corpus` is exercised only by the manual `scripts/smoke_test_pipeline.py`, which Task 3 below runs). Correctness here is verified by re-running the full existing pytest suite (nothing in it imports `run_pipeline`, so this is a no-regression check) and by Task 3's smoke run.

- [ ] **Step 1: Add the import**

In `src/run_pipeline.py`, alongside the existing `from src.models_stage2 import train_stage2`:

```python
from src.models_flat import train_flat
```

- [ ] **Step 2: Train the flat baseline in `run_corpus`, right after the Stage 2 block and before `real_synth_df` is built**

Insert after the line `_log(f"[{name}] Stage 2 macro_f1={stage2_metrics['macro_f1']:.4f}")` (currently `src/run_pipeline.py:171`):

```python
    _log(f"[{name}] Flat baseline: training models {model_names or 'ALL'} on all {len(train_df):,} rows (24-class label)")
    t0 = time.time()
    flat = train_flat(
        train_df, val_df, numeric_cols=numeric_cols, categorical_cols=CATEGORICAL_COLS,
        model_names=model_names,
    )
    _log(f"[{name}] Flat baseline done in {time.time() - t0:.1f}s")
    flat_val_preds = flat["models"]["lightgbm"]["val_predictions"]

    flat_24class_metrics = subclass_metrics(flat["class_true"], flat_val_preds)
    _log(f"[{name}] Flat 24-class macro_f1={flat_24class_metrics['macro_f1']:.4f}")

    # Same rows Stage 1 sees (all of val_df) and the same ground-truth family
    # column Stage 1 trains against — collapse the flat model's raw 24-class
    # predictions to family via the one true class->family map, so this is
    # directly comparable to stage1_metrics.macro_f1.
    flat_family_true = val_df[FAMILY_LABEL_COL].values
    flat_family_pred = derive_family_label(pd.Series(flat_val_preds)).values
    flat_family_metrics = family_metrics(flat_family_true, flat_family_pred)
    _log(f"[{name}] Flat family-derived macro_f1={flat_family_metrics['macro_f1']:.4f}")

    # Same oracle-family-filtered subset Stage 2 sees — directly comparable
    # to stage2_metrics.macro_f1.
    flat_pq_hybrid_mask = pd.Series(flat_family_true).isin(["Post-Quantum", "Hybrid"]).values
    flat_subtype_metrics = subclass_metrics(
        flat["class_true"][flat_pq_hybrid_mask], flat_val_preds[flat_pq_hybrid_mask]
    )
    _log(f"[{name}] Flat subtype(PQ/Hybrid subset) macro_f1={flat_subtype_metrics['macro_f1']:.4f}")

    flat_metrics = {
        "flat_24class": flat_24class_metrics,
        "flat_family_derived": flat_family_metrics,
        "flat_subtype_pq_hybrid": flat_subtype_metrics,
    }
```

- [ ] **Step 3: Write the new results file and extend the return value**

Right after the existing block that writes `stage1_metrics`/`stage2_metrics`/`real_vs_synth` (currently `src/run_pipeline.py:178-182`), add:

```python
    (RESULTS_DIR / f"{name}_flat_metrics.json").write_text(json.dumps(_json_safe(flat_metrics), indent=2))
    _log(f"[{name}] wrote results/{name}_flat_metrics.json")
```

Then change the function's `return` statement (currently `src/run_pipeline.py:184`) from:

```python
    return {"stage1": stage1_metrics, "stage2": stage2_metrics}
```

to:

```python
    return {"stage1": stage1_metrics, "stage2": stage2_metrics, "flat": flat_metrics}
```

- [ ] **Step 4: Extend the `comparison` dict in `__main__`**

In the `comparison = {...}` block (currently `src/run_pipeline.py:223-228`), add six keys so the headline hierarchical-vs-flat numbers land in `csv_vs_h5_comparison.json`:

```python
    comparison = {
        "csv_stage1_macro_f1": csv_results["stage1"]["macro_f1"],
        "h5_stage1_macro_f1": h5_results["stage1"]["macro_f1"],
        "csv_stage2_macro_f1": csv_results["stage2"]["macro_f1"],
        "h5_stage2_macro_f1": h5_results["stage2"]["macro_f1"],
        "csv_flat_24class_macro_f1": csv_results["flat"]["flat_24class"]["macro_f1"],
        "h5_flat_24class_macro_f1": h5_results["flat"]["flat_24class"]["macro_f1"],
        "csv_flat_family_derived_macro_f1": csv_results["flat"]["flat_family_derived"]["macro_f1"],
        "h5_flat_family_derived_macro_f1": h5_results["flat"]["flat_family_derived"]["macro_f1"],
        "csv_flat_subtype_pq_hybrid_macro_f1": csv_results["flat"]["flat_subtype_pq_hybrid"]["macro_f1"],
        "h5_flat_subtype_pq_hybrid_macro_f1": h5_results["flat"]["flat_subtype_pq_hybrid"]["macro_f1"],
    }
```

- [ ] **Step 5: Update the smoke test script's summary print for parity**

In `scripts/smoke_test_pipeline.py`, after the existing lines printing `stage1`/`stage2` macro-F1 (currently lines 69-72), add:

```python
    print("CSV flat 24class macro_f1:", csv_results["flat"]["flat_24class"]["macro_f1"])
    print("CSV flat family-derived macro_f1:", csv_results["flat"]["flat_family_derived"]["macro_f1"])
    print("H5 flat 24class macro_f1:", h5_results["flat"]["flat_24class"]["macro_f1"])
    print("H5 flat family-derived macro_f1:", h5_results["flat"]["flat_family_derived"]["macro_f1"])
```

- [ ] **Step 6: Run the full existing suite to confirm no regressions**

Run: `python -m pytest -v`
Expected: same pass count as the end of Task 1 (23 passed), 0 failures — this task touches no test files, so this is a pure regression check.

- [ ] **Step 7: Commit**

```bash
git add src/run_pipeline.py scripts/smoke_test_pipeline.py
git commit -m "feat: wire flat 24-class baseline into the real-data pipeline"
```

---

## Task 3: Smoke-test the wiring on 40k rows before the real run

**Files:** none modified — this is a verification-only task using the existing `scripts/smoke_test_pipeline.py`.

- [ ] **Step 1: Run the smoke test**

```bash
python scripts/smoke_test_pipeline.py
```

Expected: exits 0; prints, among the existing Stage 1/Stage 2 lines, four new lines (`CSV flat 24class macro_f1: ...`, `CSV flat family-derived macro_f1: ...`, `H5 flat 24class macro_f1: ...`, `H5 flat family-derived macro_f1: ...`) with values strictly between 0.0 and 1.0 (a degenerate 0.0 or 1.0 on 40k real rows would indicate the flat model saw no signal or leaked the label — investigate before proceeding rather than treating either as acceptable).

- [ ] **Step 2: Inspect one of the new results files**

```bash
python -c "import json; print(json.dumps(json.load(open('data/smoke/results/csv_flat_metrics.json')), indent=2)[:1000])"
```

Expected: valid JSON with the three top-level keys `flat_24class`, `flat_family_derived`, `flat_subtype_pq_hybrid`, each containing a `macro_f1` field.

No commit — `data/smoke/` is scratch output, not a deliverable (matches how Task 14's original smoke test was run and discarded).

---

## Task 4: Run the real 5M-row pipeline and commit results

**Files:**
- Modify (generated): `results/csv_flat_metrics.json`, `results/h5_flat_metrics.json`, `results/csv_vs_h5_comparison.json` (overwritten with the extended six new keys), plus `results/csv_stage1_metrics.json`/`csv_stage2_metrics.json`/`csv_real_vs_synthetic.json` and their `h5_` counterparts (rewritten with identical content to before, since Stage 1/Stage 2 logic is unchanged by this plan — only `run_corpus`'s new flat-baseline branch and the six added comparison keys should actually differ from the currently-committed files).

- [ ] **Step 1: Confirm the Parquet cache is still present (avoids re-reading the 8.7GB raw files)**

```bash
python -c "from src.config import CSV_CORPUS_DIR, H5_CORPUS_DIR; from pathlib import Path; print((CSV_CORPUS_DIR / 'data.parquet').exists(), (H5_CORPUS_DIR / 'data.parquet').exists())"
```

Expected: `True True`. If either is `False`, the run will re-convert that corpus from the raw Drive-hosted file first (adds time; not an error).

- [ ] **Step 2: Run the full pipeline in the background**

```bash
python -m src.run_pipeline
```

Expected runtime: the flat baseline trains LightGBM on the same ~3.5M rows Stage 1 already trains on, so budget roughly the same per-corpus time Stage 1 already takes (well under an hour per corpus per the Task 14 timing note) — run in the background and check periodically rather than blocking on a single call.

- [ ] **Step 3: Inspect the results**

```bash
python -c "
import json
from pathlib import Path
for p in sorted(Path('results').glob('*flat*.json')) + [Path('results/csv_vs_h5_comparison.json')]:
    print('---', p.name, '---')
    print(json.dumps(json.load(open(p)), indent=2)[:600])
"
```

Confirm: `csv_flat_metrics.json`/`h5_flat_metrics.json` each have non-degenerate `flat_24class`/`flat_family_derived`/`flat_subtype_pq_hybrid` blocks (macro_f1 strictly between 0 and 1, `confusion_matrix`/`per_class` not empty); `csv_vs_h5_comparison.json` has all ten macro-F1 keys (four pre-existing, six new).

- [ ] **Step 4: Commit the results**

```bash
git add results/*.json
git commit -m "feat: add flat 24-class baseline results from the real 5M-row run"
```

---

## Task 5: Update STATUS.md and PLANO_DE_ESTUDO.md

**Files:**
- Modify: `STATUS.md`
- Modify: `PLANO_DE_ESTUDO.md`

- [ ] **Step 1: Move STATUS.md's item 1 out of "O que falta decidir" into a new results row**

Read the actual `macro_f1` values written in Task 4 from `results/csv_flat_metrics.json`/`results/h5_flat_metrics.json` and:
- Add a row to the "Resultados obtidos" table (or a new table directly below it) with the flat-baseline numbers next to the existing Stage 1/Stage 2 rows, using the same CSV/H5 column layout already in that table (`STATUS.md:40-43`).
- Remove item 1 (`STATUS.md:47-49`) from "O que falta decidir", renumbering the remaining two items.
- Add a line to "Log de execuções" (`STATUS.md:61-64`) noting the date and that this run added the flat 24-class baseline, pointing at the new `results/*_flat_metrics.json` files.

- [ ] **Step 2: Record the outcome in PLANO_DE_ESTUDO.md**

Add a short "Task 14 Amendment 3" section after the existing Amendment 2 (after `PLANO_DE_ESTUDO.md:1738`, the line that listed the flat baseline as out of scope) stating it has now been implemented, with a one-line pointer to this plan file and the resulting macro-F1 numbers.

- [ ] **Step 3: Commit**

```bash
git add STATUS.md PLANO_DE_ESTUDO.md
git commit -m "docs: record flat 24-class baseline results, close STATUS.md gap 1"
```
