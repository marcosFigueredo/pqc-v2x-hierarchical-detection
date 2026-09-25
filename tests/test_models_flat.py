import numpy as np
import pandas as pd

from src.evaluate import subclass_metrics
from src.models_flat import flat_comparison_metrics, train_flat


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
        assert res["eval_predictions"].shape[0] == len(val_df)


def test_train_flat_uses_class_label_by_default():
    train_df = _synthetic_df(200)
    val_df = _synthetic_df(80)

    result = train_flat(train_df, val_df, numeric_cols=["num_feat"], categorical_cols=["cat_feat"])

    assert set(np.unique(result["class_true"])) <= set(val_df["class_label"].unique())


def test_flat_comparison_metrics_subtype_mask_uses_true_family_not_predicted():
    # A PQ/Hybrid row (by TRUE family) whose flat-model family, once derived
    # from its class prediction, lands in a DIFFERENT family must still be
    # included in the subtype comparison -- filtering must key off the true
    # family column, never the flat model's own (possibly wrong) predicted
    # family, or Stage-2-comparable rows would silently go missing.
    class_true = np.array([1, 3, 17])
    class_pred = np.array([1, 3, 17])
    family_true = np.array(["Post-Quantum", "Hybrid", "Classical"])
    # Every predicted family disagrees with the true family -- if the mask
    # used family_pred instead of family_true, the PQ/Hybrid subset would be
    # empty here.
    family_pred = np.array(["Classical", "Classical", "Post-Quantum"])

    result = flat_comparison_metrics(class_true, class_pred, family_true, family_pred)
    subtype = result["flat_subtype_pq_hybrid"]

    # Only the two TRUE PQ/Hybrid rows (class 1, class 3) belong here --
    # class 17's row is truly Classical and must be excluded regardless of
    # what family_pred says about the other two.
    assert subtype["labels"] == [1, 3]
    assert set(subtype["per_class"].keys()) == {1, 3}


def test_flat_comparison_metrics_subtype_label_set_matches_true_labels_only():
    # Regression test for the Finding-1 bug: the flat model (trained on all
    # 24 classes) can predict a class that never truly occurs in the
    # PQ/Hybrid-filtered subset (here, 99) -- a family-restricted model like
    # Stage 2 structurally cannot. Averaging over the union of true/pred
    # labels (the old, buggy behavior) folds that phantom label into
    # macro_f1 with zero true instances, unfairly dragging the flat model's
    # score down relative to Stage 2. The fair comparison space is the TRUE
    # label set alone.
    class_true = np.array([1, 1, 2, 2])
    class_pred = np.array([1, 2, 2, 99])  # 99 never truly occurs here
    family_true = np.array(["Post-Quantum", "Post-Quantum", "Hybrid", "Hybrid"])
    family_pred = np.array(["Post-Quantum", "Post-Quantum", "Hybrid", "Hybrid"])

    result = flat_comparison_metrics(class_true, class_pred, family_true, family_pred)
    subtype = result["flat_subtype_pq_hybrid"]

    assert subtype["labels"] == [1, 2]
    assert 99 not in subtype["per_class"]

    # A same-shape "stage2-like" metric computed directly with the true
    # label set must match exactly -- same label set, same macro_f1.
    stage2_like = subclass_metrics(class_true, class_pred, labels=[1, 2])
    assert subtype["macro_f1"] == stage2_like["macro_f1"]

    # And it must differ from the naive union-based score (what the old
    # code produced), which is what this test would have caught.
    naive = subclass_metrics(class_true, class_pred)
    assert naive["labels"] == [1, 2, 99]
    assert subtype["macro_f1"] > naive["macro_f1"]
