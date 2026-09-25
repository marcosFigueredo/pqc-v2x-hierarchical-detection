import numpy as np
import pandas as pd

from src.models_stage1 import train_stage1


def _synthetic_df(n):
    rng = np.random.default_rng(0)
    class_label = rng.integers(0, 4, size=n)
    family_map = {0: "Normal", 1: "Post-Quantum", 2: "Hybrid", 3: "Classical"}
    return pd.DataFrame(
        {
            "class_label": class_label,
            "family": [family_map[c] for c in class_label],
            "num_feat": class_label + rng.normal(0, 0.5, size=n),
            "cat_feat": rng.choice(["urban", "rural"], size=n),
        }
    )


def test_train_stage1_returns_predictions_for_all_val_rows():
    train_df = _synthetic_df(300)
    val_df = _synthetic_df(100)

    result = train_stage1(
        train_df, val_df, numeric_cols=["num_feat"], categorical_cols=["cat_feat"],
        label_col="family",
    )

    assert len(result["family_true"]) == len(val_df)
    for name, res in result["models"].items():
        assert res["eval_predictions"].shape[0] == len(val_df)
