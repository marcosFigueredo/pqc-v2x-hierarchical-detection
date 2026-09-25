"""Guard for Amendment 3 / B1: reported metrics must come from the TEST split.

Until 2026-09-23 the pipeline trained on `train_df` and then scored every
model on `val_df`, while the manuscript stated that all metrics came from the
test partition and never from validation. `test_df` was split, checked for
group leakage, and then never read again. Nothing was tuned on validation, so
the numbers were not inflated — but the claim was false, and the partition had
been reused across months of iteration. These tests fail if the evaluation
partition ever silently moves back.
"""
import numpy as np
import pandas as pd
import pytest

import src.run_pipeline as pipeline
from src.run_pipeline import CATEGORICAL_COLS, NUMERIC_COLS


def _corpus(n=4000, seed=0):
    """A small corpus with every column the pipeline expects as an input."""
    rng = np.random.default_rng(seed)
    # 24 subtypes so derive_family_label produces all four families.
    class_label = rng.integers(0, 24, size=n)
    frame = {"class_label": class_label}
    for col in NUMERIC_COLS:
        # The signal has to be strong enough that Stage 1 actually routes
        # some rows to Stage 2 — with pure noise the routed set comes back
        # empty and the cascade metrics have no labels to average over.
        # The test asserts plumbing, not accuracy, but it needs a cascade
        # that runs end to end.
        frame[col] = class_label * 2.0 + rng.normal(0, 0.3, size=n)
    for col in CATEGORICAL_COLS:
        frame[col] = rng.choice(["a", "b", "c"], size=n)
    # `is_real_trace` sits in NUMERIC_COLS but is a genuine bool in both
    # corpora, and the real-versus-synthetic split compares it against True.
    # Leaving it as a float makes that comparison match nothing.
    frame["is_real_trace"] = rng.choice([True, False], size=n)
    return pd.DataFrame(frame)


@pytest.fixture
def corpus_parquet(tmp_path):
    path = tmp_path / "data.parquet"
    _corpus().to_parquet(path)
    return path


def test_metrics_are_computed_on_the_test_partition(corpus_parquet, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path / "results")
    result = pipeline.run_corpus(
        "unit", corpus_parquet, [], model_names=["lightgbm"], seed=42, bootstrap=False,
    )

    provenance = result["provenance"]
    assert provenance["evaluation_partition"] == "test"
    # The three partitions must be disjoint and the evaluated one must be the
    # test split, not the validation split.
    assert provenance["n_eval_rows"] > 0
    assert provenance["n_val_rows_held_out_unused"] > 0
    assert (
        provenance["n_train_rows"]
        + provenance["n_eval_rows"]
        + provenance["n_val_rows_held_out_unused"]
        == 4000
    )


def test_every_metric_block_carries_its_partition(corpus_parquet, tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(pipeline, "RESULTS_DIR", results_dir)
    pipeline.run_corpus(
        "unit", corpus_parquet, [], model_names=["lightgbm"], seed=42, bootstrap=False,
    )

    import json
    for filename in (
        "unit_stage1_metrics.json",
        "unit_stage2_metrics.json",
        "unit_flat_metrics.json",
        "unit_real_cascade_metrics.json",
    ):
        written = json.loads((results_dir / filename).read_text())
        assert written["provenance"]["evaluation_partition"] == "test", filename


def test_trivial_floors_are_reported_for_every_task(corpus_parquet, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path / "results")
    result = pipeline.run_corpus(
        "unit", corpus_parquet, [], model_names=["lightgbm"], seed=42, bootstrap=False,
    )

    for task in ("stage1", "stage2", "flat"):
        floors = result["trivial_baselines"][task]
        assert 0.0 <= floors["majority_class"]["macro_f1"] <= 1.0
        assert 0.0 <= floors["stratified_random"]["macro_f1"] <= 1.0


def test_end_to_end_keeps_every_true_pq_hybrid_row(corpus_parquet, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path / "results")
    result = pipeline.run_corpus(
        "unit", corpus_parquet, [], model_names=["lightgbm"], seed=42, bootstrap=False,
    )

    cascade = result["real_cascade"]
    stage2_e2e = cascade["stage2_end_to_end"]
    flat_e2e = cascade["flat_end_to_end"]

    # Both architectures must be scored over the identical row set, and that
    # set must be every truly PQ/Hybrid row — including the ones Stage 1
    # never routed, which the routed-only metric drops.
    assert stage2_e2e["n_total"] == flat_e2e["n_total"]
    assert flat_e2e["n_never_routed"] == 0
    assert (
        stage2_e2e["n_routed"] + stage2_e2e["n_never_routed"] == stage2_e2e["n_total"]
    )
    assert (
        stage2_e2e["n_never_routed"]
        == cascade["routing_diagnostics"]["not_routed_but_truly_pq_hybrid"]
    )
