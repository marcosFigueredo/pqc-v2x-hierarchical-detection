# Real Stage1→Stage2 Cascade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the optimistic, oracle-family-conditioned Stage 2 metric with a metric computed using Stage 1's own real predicted routing, and produce a symmetric flat-model comparison on the exact same routed rows — closing STATUS.md's gap 1 ("Cascata real Stage1→Stage2") and giving Hypothesis 1's subtype-level claim its most rigorous possible evidence: a real (non-oracle) hierarchical pipeline vs. the flat baseline, evaluated on identical rows.

**Architecture:** Extend `train_stage2` with an optional `predict_df` parameter: after fitting on the true-PQ/Hybrid-filtered training rows exactly as today, also transform and predict on whatever rows the caller passes in — using the SAME fitted preprocessor, never refit — and return those predictions alongside the existing oracle-conditioned output. In `run_pipeline.py::run_corpus`, build the routed subset from Stage 1's own predictions (`val_df` rows where Stage 1 predicted "Post-Quantum" or "Hybrid" — regardless of true family), feed it to `train_stage2` as `predict_df`, and score the result against ground truth. Because this routed subset can contain false positives (Stage 1 wrongly routed a Normal/Classical row) and miss false negatives (a true PQ/Hybrid row Stage 1 routed elsewhere), it is a structurally different, harder evaluation than the existing oracle-conditioned `stage2_metrics` — report it as a new, separate number, not a replacement. For the comparison to remain fair to Hypothesis 1, also score the flat model's predictions on this *exact same* routed subset (same rows, same restricted label set, using the same true-label-set-restriction technique the flat-baseline plan already established), so both the real hierarchical pipeline and the flat baseline are judged on identical ground.

**Tech Stack:** Python, pandas, numpy, scikit-learn, LightGBM — same stack as the rest of `src/`.

**Spec:** [STATUS.md](../../../STATUS.md) gap 1 under "O que falta decidir" (the gap this plan closes) and [docs/superpowers/specs/2026-08-26-pqc-hierarchical-detection-design.md](../specs/2026-08-26-pqc-hierarchical-detection-design.md) — Data Strategy section: "Stage 1's predicted routing is used only at evaluation time, to measure realistic end-to-end pipeline error" — and Evaluation section: "End-to-end: pipeline metric using Stage-1-predicted routing feeding Stage 2 (measures realistic compounded error)". Also see [docs/superpowers/plans/2026-08-28-flat-24class-baseline.md](2026-08-28-flat-24class-baseline.md) (the immediately prior plan) for the label-set-restriction technique (`flat_comparison_metrics` in `src/models_flat.py`) this plan reuses.

## Global Constraints

- Reuse `SEED = 42` from `src/config.py` — never hardcode a different seed.
- Reuse the existing `_dedup_group`-based split and the existing `train_df`/`val_df` already computed once per corpus in `run_corpus` — never re-split, never retrain Stage 1/Stage 2 differently than today (this plan changes what gets *evaluated*, not how models are *trained*: Stage 2 keeps training on ground-truth-filtered PQ/Hybrid rows, per the spec — "Stage 2 training data: PQ+Hybrid rows only ... using ground-truth family labels for the training set ... Stage 1's predicted routing is used only at evaluation time").
- The real run trains **LightGBM only** (`REAL_RUN_MODEL_NAMES = ["lightgbm"]`) — all new code must thread `model_names`/read predictions keyed by `"lightgbm"`, never hardcode a different model list.
- Reuse the true-label-set restriction technique from `src/models_flat.py::flat_comparison_metrics` (label set = the labels that actually occur in the TRUE labels of the subset being scored, never derived from predictions) for any comparison between the flat model and a hierarchical stage evaluated on the same rows — this is the exact bug the previous plan's final review caught and fixed; do not reintroduce a union-of-true-and-predicted label set for a paired comparison.
- `subclass_metrics` (`src/evaluate.py`) already accepts an optional `labels=` parameter — reuse it; do not add a second, parallel metrics function.
- Existing behavior must not change when `predict_df` is not passed: `train_stage2(train_df, val_df, ...)` with no `predict_df` argument must return exactly what it returns today (backward compatible — the existing test `test_train_stage2_only_uses_pq_hybrid_rows` must keep passing unmodified).

---

## Task 1: `train_stage2` real-routing prediction + `routing_diagnostics`

**Files:**
- Modify: `src/models_stage2.py`
- Test: `tests/test_models_stage2.py`

**Interfaces:**
- Consumes: existing `add_missingness_indicators`, `build_preprocessor` (`src/features.py`); existing `train_and_eval_models` (`src/model_training.py`).
- Produces:
  - `train_stage2(train_df, val_df, numeric_cols, categorical_cols, family_col=FAMILY_LABEL_COL, subclass_col="class_label", pq_hybrid_families=("Post-Quantum", "Hybrid"), model_names=None, predict_df=None)` — when `predict_df` is `None` (default), returns exactly `{"subclass_true": ..., "models": ...}` as today. When `predict_df` is a DataFrame, the returned dict additionally has `"routed_true": ndarray` (== `predict_df[subclass_col].values`, in `predict_df`'s row order) and `"routed_predictions": {name: ndarray}` (one array per trained model, same order).
  - `routing_diagnostics(true_family, predicted_family, pq_hybrid_families=("Post-Quantum", "Hybrid")) -> dict` — new pure function, also exported from `src/models_stage2.py`. Returns `{"routed_count": int, "true_pq_hybrid_count": int, "routed_and_truly_pq_hybrid": int, "routed_but_not_truly_pq_hybrid": int, "not_routed_but_truly_pq_hybrid": int, "routing_precision": float, "routing_recall": float}`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_models_stage2.py
import numpy as np
import pandas as pd

from src.models_stage2 import routing_diagnostics, train_stage2


def test_train_stage2_predict_df_returns_routed_predictions_using_fitted_preprocessor():
    train_df = _synthetic_df(400)
    val_df = _synthetic_df(150)
    # Deliberately unfiltered by family (unlike train_df/val_df's internal
    # PQ/Hybrid filtering) — this mimics Stage 1 routing a Normal/Classical
    # row into the subset by mistake, which train_stage2 must NOT filter out.
    predict_df = _synthetic_df(50)

    result = train_stage2(
        train_df, val_df, numeric_cols=["num_feat"], categorical_cols=["cat_feat"],
        predict_df=predict_df,
    )

    assert "routed_true" in result
    assert "routed_predictions" in result
    np.testing.assert_array_equal(result["routed_true"], predict_df["class_label"].values)
    for name, preds in result["routed_predictions"].items():
        assert preds.shape[0] == len(predict_df)


def test_train_stage2_without_predict_df_omits_routed_keys():
    train_df = _synthetic_df(200)
    val_df = _synthetic_df(80)

    result = train_stage2(train_df, val_df, numeric_cols=["num_feat"], categorical_cols=["cat_feat"])

    assert "routed_true" not in result
    assert "routed_predictions" not in result


def test_routing_diagnostics_counts_tp_fp_fn_and_rates():
    true_family = np.array(["Post-Quantum", "Post-Quantum", "Normal", "Hybrid", "Classical"])
    pred_family = np.array(["Post-Quantum", "Normal", "Post-Quantum", "Hybrid", "Classical"])
    # routed (predicted PQ/Hybrid): idx 0,2,3 -> 3
    # truly PQ/Hybrid: idx 0,1,3 -> 3
    # TP (routed & truly): idx 0,3 -> 2
    # FP (routed & not truly): idx 2 -> 1
    # FN (not routed & truly): idx 1 -> 1

    result = routing_diagnostics(true_family, pred_family)

    assert result["routed_count"] == 3
    assert result["true_pq_hybrid_count"] == 3
    assert result["routed_and_truly_pq_hybrid"] == 2
    assert result["routed_but_not_truly_pq_hybrid"] == 1
    assert result["not_routed_but_truly_pq_hybrid"] == 1
    assert abs(result["routing_precision"] - 2 / 3) < 1e-9
    assert abs(result["routing_recall"] - 2 / 3) < 1e-9


def test_routing_diagnostics_handles_zero_routed_and_zero_true_pq_hybrid():
    true_family = np.array(["Normal", "Classical"])
    pred_family = np.array(["Normal", "Classical"])

    result = routing_diagnostics(true_family, pred_family)

    assert result["routed_count"] == 0
    assert result["true_pq_hybrid_count"] == 0
    assert result["routing_precision"] == 0.0
    assert result["routing_recall"] == 0.0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_models_stage2.py -v`
Expected: FAIL — `ImportError: cannot import name 'routing_diagnostics'` (and the `predict_df` tests fail with `TypeError: train_stage2() got an unexpected keyword argument 'predict_df'` once the import is worked around, but the import failure is what you'll see first since it's a module-level import).

- [ ] **Step 3: Write the minimal implementation**

Replace `src/models_stage2.py` in full with:

```python
import numpy as np

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
    model_names=None,
    predict_df=None,
):
    train_pq = train_df[train_df[family_col].isin(pq_hybrid_families)].reset_index(drop=True)
    val_pq = val_df[val_df[family_col].isin(pq_hybrid_families)].reset_index(drop=True)

    sparse_present = [c for c in SPARSE_TRACE_COLUMNS if c in train_df.columns]

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

    models = train_and_eval_models(X_train, y_train, X_val, y_val, model_names=model_names)

    result = {"subclass_true": y_val, "models": models}

    if predict_df is not None:
        # Deliberately NOT filtered by family_col — predict_df is the set of
        # rows Stage 1 *predicted* as PQ/Hybrid, which may include rows whose
        # true family is Normal/Classical (Stage 1 false positives) and may
        # be missing true PQ/Hybrid rows Stage 1 routed elsewhere (false
        # negatives). Filtering here would silently turn this back into the
        # oracle-conditioned evaluation this parameter exists to avoid.
        predict_df = predict_df.reset_index(drop=True)
        predict_df = add_missingness_indicators(predict_df, sparse_present)
        X_predict = preprocessor.transform(predict_df)
        result["routed_true"] = predict_df[subclass_col].values
        result["routed_predictions"] = {
            name: res["model"].predict(X_predict) for name, res in models.items()
        }

    return result


def routing_diagnostics(true_family, predicted_family, pq_hybrid_families=("Post-Quantum", "Hybrid")):
    true_family = np.asarray(true_family)
    predicted_family = np.asarray(predicted_family)

    truly_pq_hybrid = np.isin(true_family, list(pq_hybrid_families))
    routed = np.isin(predicted_family, list(pq_hybrid_families))

    routed_count = int(routed.sum())
    true_pq_hybrid_count = int(truly_pq_hybrid.sum())
    true_positive = int((routed & truly_pq_hybrid).sum())
    false_positive = int((routed & ~truly_pq_hybrid).sum())
    false_negative = int((~routed & truly_pq_hybrid).sum())

    return {
        "routed_count": routed_count,
        "true_pq_hybrid_count": true_pq_hybrid_count,
        "routed_and_truly_pq_hybrid": true_positive,
        "routed_but_not_truly_pq_hybrid": false_positive,
        "not_routed_but_truly_pq_hybrid": false_negative,
        "routing_precision": (true_positive / routed_count) if routed_count else 0.0,
        "routing_recall": (true_positive / true_pq_hybrid_count) if true_pq_hybrid_count else 0.0,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_models_stage2.py -v`
Expected: all pass (1 pre-existing + 4 new = 5 passed).

- [ ] **Step 5: Run the full existing suite to confirm no regressions**

Run: `python -m pytest -v`
Expected: 26 pre-existing + 4 new = 30 passed, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/models_stage2.py tests/test_models_stage2.py
git commit -m "feat: add real-routing prediction + routing diagnostics to Stage 2"
```

---

## Task 2: Wire the real cascade into `run_pipeline.py`

**Files:**
- Modify: `src/run_pipeline.py`
- Modify: `scripts/smoke_test_pipeline.py`

**Interfaces:**
- Consumes: `train_stage2`'s new `predict_df`/`"routed_true"`/`"routed_predictions"` and `routing_diagnostics` from Task 1 (`src.models_stage2`); existing `subclass_metrics` (`src.evaluate`); existing `flat_val_preds`/`flat["class_true"]` already computed in `run_corpus` by the prior plan.
- Produces: `run_corpus(...)` returns a new top-level key `"real_cascade": {"routing_diagnostics": ..., "stage2_real_routing": ..., "flat_real_routing": ...}` alongside the existing `"stage1"`/`"stage2"`/`"flat"` keys; writes a new `results/{name}_real_cascade_metrics.json` per corpus; the `__main__` block's `comparison` dict gains four new keys.

No new automated test for this task, matching the prior plan's precedent for `run_pipeline.py` wiring (it has no unit tests today; Task 3 below is the manual verification). Correctness here is verified by re-running the full pytest suite (a pure no-regression check, since nothing in it imports `run_pipeline`) and by Task 3's smoke run.

- [ ] **Step 1: Update the import**

In `src/run_pipeline.py`, change:

```python
from src.models_stage2 import train_stage2
```

to:

```python
from src.models_stage2 import routing_diagnostics, train_stage2
```

- [ ] **Step 2: Build the routed subset right after Stage 1, and pass it into Stage 2**

Replace the existing Stage 2 block:

```python
    _log(f"[{name}] Stage 2: training models {model_names or 'ALL'} on PQ/Hybrid subset")
    t0 = time.time()
    stage2 = train_stage2(
        train_df, val_df, numeric_cols=numeric_cols, categorical_cols=CATEGORICAL_COLS,
        model_names=model_names,
    )
    _log(f"[{name}] Stage 2 done in {time.time() - t0:.1f}s")
    best_stage2_preds = stage2["models"]["lightgbm"]["val_predictions"]
    stage2_metrics = subclass_metrics(stage2["subclass_true"], best_stage2_preds)
    _log(f"[{name}] Stage 2 macro_f1={stage2_metrics['macro_f1']:.4f}")
```

with:

```python
    # Stage 1's own predicted routing (not the true family) — the subset a
    # real, deployed pipeline would actually send to Stage 2. May contain
    # false positives (true family is Normal/Classical) and miss false
    # negatives (a true PQ/Hybrid row Stage 1 routed elsewhere).
    routed_mask = np.isin(best_stage1_preds, ["Post-Quantum", "Hybrid"])
    val_routed_df = val_df[routed_mask].reset_index(drop=True)

    _log(f"[{name}] Stage 2: training models {model_names or 'ALL'} on PQ/Hybrid subset")
    t0 = time.time()
    stage2 = train_stage2(
        train_df, val_df, numeric_cols=numeric_cols, categorical_cols=CATEGORICAL_COLS,
        model_names=model_names, predict_df=val_routed_df,
    )
    _log(f"[{name}] Stage 2 done in {time.time() - t0:.1f}s")
    best_stage2_preds = stage2["models"]["lightgbm"]["val_predictions"]
    stage2_metrics = subclass_metrics(stage2["subclass_true"], best_stage2_preds)
    _log(f"[{name}] Stage 2 (oracle-conditioned) macro_f1={stage2_metrics['macro_f1']:.4f}")

    routing_stats = routing_diagnostics(val_df[FAMILY_LABEL_COL].values, best_stage1_preds)
    routed_labels = sorted(set(stage2["routed_true"]))
    stage2_real_metrics = subclass_metrics(
        stage2["routed_true"], stage2["routed_predictions"]["lightgbm"], labels=routed_labels
    )
    _log(
        f"[{name}] Stage 2 REAL-routing macro_f1={stage2_real_metrics['macro_f1']:.4f} "
        f"(routed {routing_stats['routed_count']:,} rows, "
        f"routing precision={routing_stats['routing_precision']:.3f} "
        f"recall={routing_stats['routing_recall']:.3f})"
    )
```

`np` is already imported at the top of `src/run_pipeline.py` (`import numpy as np`) — no new import needed for `np.isin`.

- [ ] **Step 3: Add the symmetric flat-on-routed-subset comparison right after the flat baseline block**

Immediately after the existing block that logs `Flat subtype(PQ/Hybrid subset) macro_f1=...` (right before the `real_synth_df = val_df.iloc[...]` line), insert:

```python
    # Symmetric comparison for Hypothesis 1: score the flat model on the
    # EXACT SAME rows Stage 1 actually routed to Stage 2 (not the
    # oracle-true-family subset flat_metrics["flat_subtype_pq_hybrid"]
    # uses) — same true-label-set restriction technique as
    # flat_comparison_metrics, for the same reason: the flat model can
    # predict labels outside routed_labels, Stage 2 cannot.
    flat_routed_pred = flat_val_preds[routed_mask]
    flat_real_routing_metrics = subclass_metrics(
        stage2["routed_true"], flat_routed_pred, labels=routed_labels
    )
    _log(f"[{name}] Flat REAL-routing macro_f1={flat_real_routing_metrics['macro_f1']:.4f}")

    real_cascade_metrics = {
        "routing_diagnostics": routing_stats,
        "stage2_real_routing": stage2_real_metrics,
        "flat_real_routing": flat_real_routing_metrics,
    }
```

- [ ] **Step 4: Write the new results file and extend the return value**

Right after the existing block that writes `flat_metrics` (`(RESULTS_DIR / f"{name}_flat_metrics.json")...`), add:

```python
    (RESULTS_DIR / f"{name}_real_cascade_metrics.json").write_text(
        json.dumps(_json_safe(real_cascade_metrics), indent=2)
    )
    _log(f"[{name}] wrote results/{name}_real_cascade_metrics.json")
```

Change the function's `return` statement from:

```python
    return {"stage1": stage1_metrics, "stage2": stage2_metrics, "flat": flat_metrics}
```

to:

```python
    return {
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "flat": flat_metrics,
        "real_cascade": real_cascade_metrics,
    }
```

- [ ] **Step 5: Extend the `comparison` dict in `__main__`**

Add four keys to the existing `comparison = {...}` block:

```python
        "csv_stage2_real_routing_macro_f1": csv_results["real_cascade"]["stage2_real_routing"]["macro_f1"],
        "h5_stage2_real_routing_macro_f1": h5_results["real_cascade"]["stage2_real_routing"]["macro_f1"],
        "csv_flat_real_routing_macro_f1": csv_results["real_cascade"]["flat_real_routing"]["macro_f1"],
        "h5_flat_real_routing_macro_f1": h5_results["real_cascade"]["flat_real_routing"]["macro_f1"],
```

- [ ] **Step 6: Update the smoke test script's summary print for parity**

In `scripts/smoke_test_pipeline.py`, after the existing flat-metric print lines, add:

```python
    print("CSV stage2 REAL-routing macro_f1:", csv_results["real_cascade"]["stage2_real_routing"]["macro_f1"])
    print("CSV flat REAL-routing macro_f1:", csv_results["real_cascade"]["flat_real_routing"]["macro_f1"])
    print("H5 stage2 REAL-routing macro_f1:", h5_results["real_cascade"]["stage2_real_routing"]["macro_f1"])
    print("H5 flat REAL-routing macro_f1:", h5_results["real_cascade"]["flat_real_routing"]["macro_f1"])
```

- [ ] **Step 7: Run the full existing suite to confirm no regressions**

Run: `python -m pytest -v`
Expected: same 30 passed as the end of Task 1 — this task touches no test files, so this is a pure regression check.

- [ ] **Step 8: Commit**

```bash
git add src/run_pipeline.py scripts/smoke_test_pipeline.py
git commit -m "feat: wire real Stage1-routed cascade + symmetric flat comparison into the pipeline"
```

---

## Task 3: Smoke-test the wiring on 40k rows before the real run

**Files:** none modified — verification-only, using `scripts/smoke_test_pipeline.py`.

- [ ] **Step 1: Run the smoke test**

```bash
python scripts/smoke_test_pipeline.py
```

Expected: exits 0; prints four new lines (`CSV/H5 stage2 REAL-routing macro_f1`, `CSV/H5 flat REAL-routing macro_f1`) with values strictly between 0.0 and 1.0. Sanity-check the direction: `stage2 REAL-routing macro_f1` should be **lower** than the already-printed `CSV/H5 stage2 macro_f1` (oracle-conditioned) — a real, error-prone routing step should never look easier than an oracle-conditioned one. If it isn't lower, stop and investigate before proceeding (don't rationalize it away — this specific direction is the whole point of the fix).

- [ ] **Step 2: Inspect one of the new results files**

```bash
python -c "import json; print(json.dumps(json.load(open('data/smoke/results/csv_real_cascade_metrics.json')), indent=2)[:1200])"
```

Expected: valid JSON with top-level keys `routing_diagnostics`, `stage2_real_routing`, `flat_real_routing`; `routing_diagnostics.routed_count` should be a plausible fraction of the ~5,700-row smoke val set (not 0, not the entire set).

No commit — `data/smoke/` is scratch output, not a deliverable.

---

## Task 4: Run the real 5M-row pipeline and commit results

**Files:**
- Modify (generated): `results/csv_real_cascade_metrics.json`, `results/h5_real_cascade_metrics.json` (new), `results/csv_vs_h5_comparison.json` (extended with 4 new keys). `results/csv_stage1_metrics.json`/`csv_flat_metrics.json` and their h5 counterparts should be byte-identical to the currently-committed versions (Stage 1 and the flat model's training/predictions are untouched by this plan) — `results/csv_stage2_metrics.json`/`h5_stage2_metrics.json` should ALSO be unchanged (Stage 2's training and its oracle-conditioned val metric are untouched; only a new `predict_df` branch was added).

- [ ] **Step 1: Confirm the Parquet cache is still present**

```bash
python -c "from src.config import CSV_CORPUS_DIR, H5_CORPUS_DIR; from pathlib import Path; print((CSV_CORPUS_DIR / 'data.parquet').exists(), (H5_CORPUS_DIR / 'data.parquet').exists())"
```

Expected: `True True`.

- [ ] **Step 2: Run the full pipeline in the background**

```bash
python -m src.run_pipeline
```

Expected runtime: comparable to the flat-baseline plan's real run (~83 minutes total for both corpora) — this plan adds one cheap `predict()` call on an already-fitted Stage 2 model over a subset of val rows (a few seconds), not a new full training pass. Run in the background and check periodically.

- [ ] **Step 3: Inspect the results**

```bash
python -c "
import json
from pathlib import Path
for p in sorted(Path('results').glob('*real_cascade*.json')) + [Path('results/csv_vs_h5_comparison.json')]:
    print('---', p.name, '---')
    print(json.dumps(json.load(open(p)), indent=2)[:800])
"
```

Confirm: both `*_real_cascade_metrics.json` files have non-degenerate `stage2_real_routing`/`flat_real_routing` blocks (macro_f1 strictly between 0 and 1); `routing_diagnostics.routing_recall`/`routing_precision` are plausible (not exactly 0.0 or 1.0 at 5M-row scale); `csv_vs_h5_comparison.json` has all 14 macro-F1 keys (10 pre-existing, 4 new). Confirm `csv_stage2_real_routing_macro_f1` < `csv_stage2_macro_f1` and same for h5 — the real-routing number must be lower than the oracle-conditioned one on the full real data too, not just the smoke sample.

- [ ] **Step 4: Diff the unaffected result files to confirm no unintended drift**

```bash
git status --short results/
```

Expected: `csv_stage1_metrics.json`, `h5_stage1_metrics.json`, `csv_flat_metrics.json`, `h5_flat_metrics.json`, `csv_stage2_metrics.json`, `h5_stage2_metrics.json`, `csv_real_vs_synthetic.json`, `h5_real_vs_synthetic.json` do NOT appear as modified (only the two new `*_real_cascade_metrics.json` files and `csv_vs_h5_comparison.json` should show up). If any of those eight files show as modified, stop and investigate before committing — it would mean this plan's change had a side effect on code paths it wasn't supposed to touch.

- [ ] **Step 5: Commit the results**

```bash
git add results/csv_real_cascade_metrics.json results/h5_real_cascade_metrics.json results/csv_vs_h5_comparison.json
git commit -m "feat: add real Stage1-routed cascade results from the real 5M-row run"
```

---

## Task 5: Update STATUS.md and PLANO_DE_ESTUDO.md

**Files:**
- Modify: `STATUS.md`
- Modify: `PLANO_DE_ESTUDO.md`

- [ ] **Step 1: Close STATUS.md's gap 1**

Read the actual values written in Task 4 from `results/csv_real_cascade_metrics.json`/`results/h5_real_cascade_metrics.json` and:
- Add a new subsection (mirroring the existing "Baseline flat de 24 classes" subsection's style) presenting: the oracle-conditioned Stage 2 number (already in the table above) side by side with the new real-routing Stage 2 number and the real-routing flat number, plus the routing diagnostics (precision/recall of Stage 1's routing, as plain-language sentences a non-programmer can read — e.g. "de X linhas que o Stage 1 encaminhou para o Stage 2, Y% eram de fato PQ/Híbrido").
- Remove gap 1 ("Cascata real Stage1→Stage2") from "O que falta decidir", leaving only the "Outros modelos em escala completa" item (renumber/reword the section header, e.g. "(1 lacuna documentada, não resolvida)").
- Update "Próximo passo" to reflect only 1 remaining lacuna.
- Add a line to "Log de execuções" for this run, dated 2026-08-28, pointing at the new `results/*_real_cascade_metrics.json` files.
- Update the "Última atualização" date at the top if not already today.
- Update the "Testes automatizados" count in "O que já foi feito" to the real final count (30 + whatever Task 1 actually added, verified by running the suite — do not assume the number from this brief, confirm it).

- [ ] **Step 2: Rewrite the Hypothesis 1 narrative with the fully rigorous comparison**

In the same area of STATUS.md (and mirrored tersely in a new "Task 14 Amendment 4" section in PLANO_DE_ESTUDO.md, placed after the existing Amendment 3), state plainly:
- The real (Stage-1-routed) hierarchical pipeline's subtype macro-F1 for both corpora, compared against the flat model scored on the identical routed rows (`flat_real_routing` — NOT the oracle-conditioned `flat_subtype_pq_hybrid` number, which remains useful as the earlier, oracle-conditioned comparison but is no longer the headline).
- Whether hierarchical still wins under real routing, for both corpora, and by how much (compute this from the actual committed numbers — do not guess or copy example numbers from this brief).
- The routing diagnostics in one sentence (how much of Stage 1's imperfection propagates into this number).
- Keep it factual and terse, matching Amendment 1/2/3's style — no marketing language, no claims beyond what the numbers show.

- [ ] **Step 3: Commit**

```bash
git add STATUS.md PLANO_DE_ESTUDO.md
git commit -m "docs: record real Stage1-routed cascade results, close STATUS.md gap 1"
```
