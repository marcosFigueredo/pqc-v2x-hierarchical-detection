import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    NOT_ROUTED,
    compare_real_synthetic,
    end_to_end_subtype_metrics,
    family_metrics,
    paired_bootstrap_macro_f1,
    subclass_metrics,
    trivial_baselines,
)


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
    assert result["labels"] == [1, 2, 3]


def test_subclass_metrics_explicit_labels_restricts_averaging_scope():
    # y_pred contains a label (99) that never occurs in y_true. Default
    # (labels=None) behavior averages over the union, folding 99 in with
    # zero true instances. Passing an explicit labels list restricts the
    # macro-F1 average (and per_class/labels) to exactly that list,
    # regardless of what extra labels appear in y_pred.
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, 2, 2, 99])

    default_result = subclass_metrics(y_true, y_pred)
    assert default_result["labels"] == [1, 2, 99]

    restricted_result = subclass_metrics(y_true, y_pred, labels=[1, 2])
    assert restricted_result["labels"] == [1, 2]
    assert set(restricted_result["per_class"].keys()) == {1, 2}
    assert restricted_result["macro_f1"] > default_result["macro_f1"]


def test_compare_real_synthetic_splits_by_flag():
    df = pd.DataFrame({
        "y_true": ["Normal", "Normal", "Hybrid", "Hybrid"],
        "y_pred": ["Normal", "Hybrid", "Hybrid", "Hybrid"],
        "is_real_trace": [True, True, False, False],
    })
    result = compare_real_synthetic(df, "y_true", "y_pred")
    assert result["real"]["accuracy"] == 0.5
    assert result["synthetic"]["accuracy"] == 1.0


def test_end_to_end_charges_the_cascade_for_rows_it_never_routed():
    # Four truly PQ/Hybrid rows; Stage 1 routed only the first two, and
    # Stage 2 labelled both of those correctly. The routed-only view would
    # call this perfect; the end-to-end view must not, because two real
    # attacks never reached the detector at all.
    true_subtypes = np.array([5, 6, 7, 8])
    routed_mask = np.array([True, True, False, False])
    routed_predictions = np.array([5, 6])

    routed_only = subclass_metrics(true_subtypes[routed_mask], routed_predictions)
    end_to_end = end_to_end_subtype_metrics(true_subtypes, routed_mask, routed_predictions)

    assert routed_only["macro_f1"] == 1.0
    assert end_to_end["macro_f1"] < 1.0
    assert end_to_end["n_total"] == 4
    assert end_to_end["n_routed"] == 2
    assert end_to_end["n_never_routed"] == 2
    # The two dropped classes must show zero recall, not be silently absent.
    assert end_to_end["per_class"][7]["recall"] == 0.0
    assert end_to_end["per_class"][8]["recall"] == 0.0


def test_end_to_end_with_all_rows_routed_matches_plain_subclass_metrics():
    # A flat classifier never abstains, so its end-to-end score must reduce
    # exactly to scoring it over the same rows.
    true_subtypes = np.array([5, 6, 7, 5])
    predictions = np.array([5, 7, 7, 6])
    all_routed = np.ones(4, dtype=bool)

    end_to_end = end_to_end_subtype_metrics(true_subtypes, all_routed, predictions)
    plain = subclass_metrics(true_subtypes, predictions, labels=sorted(set(true_subtypes.tolist())))

    assert end_to_end["macro_f1"] == plain["macro_f1"]
    assert end_to_end["n_never_routed"] == 0


def test_cascade_sentinel_never_counts_as_a_correct_prediction():
    # Every row dropped: nothing can be right.
    true_subtypes = np.array([1, 2, 3])
    none_routed = np.zeros(3, dtype=bool)
    metrics = end_to_end_subtype_metrics(true_subtypes, none_routed, np.array([], dtype=int))
    assert metrics["macro_f1"] == 0.0
    assert NOT_ROUTED not in metrics["labels"]


def test_end_to_end_rejects_misaligned_mask():
    with pytest.raises(ValueError):
        end_to_end_subtype_metrics(np.array([1, 2, 3]), np.array([True, False]), np.array([1]))


def test_trivial_baselines_report_majority_and_random_floor():
    # 80/20 imbalance: the majority rule gets high accuracy but poor macro-F1,
    # which is exactly the gap these floors exist to make visible.
    y_train = np.array([0] * 80 + [1] * 20)
    y_eval = np.array([0] * 8 + [1] * 2)
    floors = trivial_baselines(y_train, y_eval, seed=0)

    assert floors["majority_class"]["label"] == 0
    assert floors["majority_class"]["accuracy"] == 0.8
    assert floors["majority_class"]["macro_f1"] < 0.5
    assert 0.0 <= floors["stratified_random"]["macro_f1"] <= 1.0


def test_paired_bootstrap_brackets_a_clear_difference():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=400)
    good = y_true.copy()
    good[:40] = (good[:40] + 1) % 4          # 10% wrong
    bad = y_true.copy()
    bad[:200] = (bad[:200] + 1) % 4          # 50% wrong

    result = paired_bootstrap_macro_f1(y_true, good, bad, n_boot=200, seed=0)

    assert result["observed_difference"] > 0
    assert result["ci95_lower"] > 0            # interval excludes "no difference"
    assert result["fraction_a_greater"] == 1.0
    assert result["n_rows"] == 400


def test_paired_bootstrap_interval_straddles_zero_for_identical_models():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, size=300)
    preds = y_true.copy()
    preds[:60] = (preds[:60] + 1) % 3

    result = paired_bootstrap_macro_f1(y_true, preds, preds, n_boot=100, seed=1)

    assert result["observed_difference"] == 0.0
    assert result["ci95_lower"] == 0.0 and result["ci95_upper"] == 0.0
