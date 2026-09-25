import numpy as np
import pandas as pd

from src.models_stage2 import train_stage2


def _synthetic_df(n):
    rng = np.random.default_rng(1)
    family = rng.choice(["Normal", "Post-Quantum", "Hybrid", "Classical"], size=n)
    subclass = np.where(
        family == "Post-Quantum", rng.integers(1, 5, size=n),
        np.where(family == "Hybrid", rng.integers(11, 15, size=n), 0),
    )
    return pd.DataFrame(
        {
            "attack_family": family,
            "class_label": subclass,
            "num_feat": subclass + rng.normal(0, 0.5, size=n),
            "cat_feat": rng.choice(["urban", "rural"], size=n),
        }
    )


def test_train_stage2_only_uses_pq_hybrid_rows():
    train_df = _synthetic_df(400)
    val_df = _synthetic_df(150)

    result = train_stage2(train_df, val_df, numeric_cols=["num_feat"], categorical_cols=["cat_feat"])

    expected_val_n = (val_df["attack_family"].isin(["Post-Quantum", "Hybrid"])).sum()
    assert len(result["subclass_true"]) == expected_val_n
    for name, res in result["models"].items():
        assert res["eval_predictions"].shape[0] == expected_val_n


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
    from src.models_stage2 import routing_diagnostics

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
    from src.models_stage2 import routing_diagnostics

    true_family = np.array(["Normal", "Classical"])
    pred_family = np.array(["Normal", "Classical"])

    result = routing_diagnostics(true_family, pred_family)

    assert result["routed_count"] == 0
    assert result["true_pq_hybrid_count"] == 0
    assert result["routing_precision"] == 0.0
    assert result["routing_recall"] == 0.0
