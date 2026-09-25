import json
import time

import numpy as np
import pandas as pd

from src.config import CSV_CORPUS_DIR, H5_CORPUS_DIR, RAW_CSV_PATH, RAW_H5_PATH, RESULTS_DIR, SEED
from src.data_prep import csv_to_parquet, h5_to_parquet, stratified_split
from src.evaluate import (
    cascade_predictions,
    compare_real_synthetic,
    end_to_end_subtype_metrics,
    family_metrics,
    paired_bootstrap_macro_f1,
    subclass_metrics,
    trivial_baselines,
)
from src.features import LEAKY_COLUMNS, derive_family_label
from src.models_flat import flat_comparison_metrics, train_flat
from src.models_stage1 import FAMILY_LABEL_COL, train_stage1
from src.models_stage2 import routing_diagnostics, train_stage2

# Amendment 2 (2026-08-27, final whole-branch review, C1): both corpora are
# NOT 5,000,000 independent observations. pq_v2x_realistic.csv has exactly
# 100,000 unique combinations of every model-input feature column, each
# replicated exactly 50 times (session_id is a 1:1 proxy for this, so
# grouping by it, as this file previously did, happened to be exactly
# correct for CSV by coincidence). H5 has an analogous ~105,000-unique-row
# duplicate structure but no usable key exposing it (session_id is a
# constant 0.0 placeholder there — see Amendment 1), so the previous
# plain-stratified fallback for H5 scattered each duplicate block's 50
# copies across train/val/test: 99.9785% of H5 validation rows had an exact
# feature-twin in training, inflating H5's measured macro-F1. Fixed by
# replacing the session_id-based (CSV-only) / no-grouping (H5) split with a
# per-row hash of the actual feature columns, computed identically for both
# corpora — see the `_dedup_group` column built in `run_corpus` below.

# Explicit, reproducible model roster for the full run.  Keep this list in
# source control (rather than relying on an ambient environment variable) so
# a result can always be traced to the exact models that were trained.
#
# The order is also intentional: LightGBM remains the primary model used for
# the historical headline fields, while the per-model fields below preserve
# the complete comparison for the remaining baselines.
REAL_RUN_MODEL_NAMES = ["lightgbm", "logreg", "random_forest", "mlp"]

# Columns present in both corpora used as model inputs (everything else in
# each corpus stays out per LEAKY_COLUMNS / is only in one corpus).
#
# Amendment 2 (2026-08-27, final whole-branch review, C2): expanded to
# include the 15 sparse crypto-trace columns (see SPARSE_TRACE_COLUMNS) and
# is_real_trace, which previously never reached the model at all. Dtypes
# verified against the real corpus before writing this list:
# protocol_mismatch/zone_mismatch_flag are object (True/False/NaN, not a
# clean numeric dtype — categorical is correct), cipher_suite/block_mode are
# low-cardinality strings (2-3 values), the other numeric sparse fields are
# float64, is_real_trace is bool with zero nulls. pqc_public_key/
# pqc_ciphertext/pqc_signature are deliberately NOT added to either list —
# they're raw cryptographic blob strings (almost-unique per non-null row),
# so only their presence/absence (via the _is_missing indicators built in
# models_stage1.py/models_stage2.py) is meaningful, never their raw value.
NUMERIC_COLS = [
    "speed", "acceleration", "direction", "snr", "packet_loss", "latency",
    "ber", "KeyReuseCount", "EncapTimeDeviation", "EntropyDeviation",
    "KeyGuessAttempts", "SignatureSize", "SpoofCertMatch", "TimingVariance",
    "LeakageSignal",
    "sequence_gap", "timestamp_offset", "message_frequency", "source_entropy",
    "handshake_time", "encryption_time", "sig_generation_time",
    "sig_verification_time", "is_real_trace",
]
CATEGORICAL_COLS = [
    "rsu_zone", "msg_type", "source", "vehicle_type", "protocol",
    "protocol_mismatch", "zone_mismatch_flag", "cipher_suite", "block_mode",
]
H5_EXTRA_NUMERIC_COLS = [
    "ZoneRisk_tunnel", "cos_timestamp", "sin_timestamp", "decryption_time",
    "enc_dec_ratio", "latency_ber_ratio", "risk_amp", "snr_packet_loss_ratio",
]


def _log(msg):
    # Operational addition (not in the brief's literal listing) so a
    # multi-hour/real background run is observable while it's happening;
    # does not change any path, function call, or JSON output.
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _json_safe(obj):
    # subclass_metrics's per_class dict has numpy.int64 keys (from the
    # class labels), which json.dumps rejects on this platform/Python
    # build. Recursively normalize numpy scalar keys/values to native
    # Python types before serializing.
    if isinstance(obj, dict):
        return {
            (k.item() if isinstance(k, np.generic) else k): _json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _metrics_by_model(model_results, metric_fn):
    """Evaluate every trained model while preserving its prediction key."""
    return {
        name: metric_fn(result["eval_predictions"])
        for name, result in model_results.items()
    }


def _convergence_by_model(model_results):
    return {
        name: result.get("convergence")
        for name, result in model_results.items()
        if result.get("convergence") is not None
    }


def run_corpus(
    name, parquet_path, extra_numeric_cols, model_names=None, seed=SEED,
    bootstrap=True, n_boot=1000, grouped_split=True,
):
    _log(f"[{name}] reading parquet from {parquet_path}")
    t0 = time.time()
    df = pd.read_parquet(parquet_path)
    _log(f"[{name}] loaded {len(df):,} rows, {df.shape[1]} cols in {time.time() - t0:.1f}s")

    # The raw attack_family column is NaN for every Normal row and for ~4%
    # of attack rows (confirmed against the real CSV during plan review) —
    # not a clean 4-class label. Overwrite it with the derived version,
    # which is built from class_label (100% populated) and always agrees
    # with the README's documented family distribution.
    df[FAMILY_LABEL_COL] = derive_family_label(df["class_label"])

    # class_label and attack_family(now derived) are labels, not features —
    # kept for splitting/training targets, even though they're in
    # LEAKY_COLUMNS as inputs to the *feature matrix* built inside
    # train_stage1/train_stage2.
    label_cols = {FAMILY_LABEL_COL, "class_label"}
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns and c not in label_cols])

    numeric_cols = NUMERIC_COLS + [c for c in extra_numeric_cols if c in df.columns]

    # Fix 1 (Amendment 2, C1): group by a hash of the row's own feature
    # values, not by a corpus-specific proxy column. This is the direct
    # generalization of what grouping by CSV's session_id was already doing
    # by coincidence, and it works identically for H5, which has no usable
    # session key at all.
    df["_dedup_group"] = pd.util.hash_pandas_object(
        df[numeric_cols + CATEGORICAL_COLS], index=False
    )

    # Amendment 3 (M3): `grouped_split=False` reproduces the protocol the
    # original PQ-V2X benchmark used — a plain stratified random split with
    # no duplicate control. It exists so the leakage claim can be shown as a
    # controlled ablation (same data, same model, only the split changes)
    # instead of asserted against a number published under a different setup.
    split_kind = "grouped by _dedup_group" if grouped_split else "PLAIN RANDOM (ablation)"
    _log(f"[{name}] splitting ({split_kind})")
    t0 = time.time()
    train_df, val_df, test_df = stratified_split(
        df, label_col="class_label",
        group_col="_dedup_group" if grouped_split else None,
        seed=seed,
    )
    _log(
        f"[{name}] split done in {time.time() - t0:.1f}s: "
        f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
    )

    train_groups = set(train_df["_dedup_group"])
    overlap = (
        (train_groups & set(val_df["_dedup_group"]))
        | (train_groups & set(test_df["_dedup_group"]))
        | (set(val_df["_dedup_group"]) & set(test_df["_dedup_group"]))
    )
    if grouped_split:
        # Safety net: confirm the property the whole grouped-split mechanism
        # exists to guarantee. Cheap relative to training; keep it
        # permanently, not just for this one run.
        assert not overlap, (
            f"{name}: {len(overlap)} duplicate-feature groups leaked across splits"
        )
        _log(f"[{name}] overlap assertion passed: no duplicate-feature groups span splits")
        leakage_rate = 0.0
    else:
        # Under the ablation the leakage is the point, so measure it rather
        # than assert it away: what fraction of evaluation rows have an exact
        # feature-twin sitting in training?
        leakage_rate = float(test_df["_dedup_group"].isin(train_groups).mean())
        _log(
            f"[{name}] PLAIN RANDOM split: {len(overlap):,} feature groups span "
            f"partitions; {leakage_rate:.4%} of test rows have an exact "
            f"feature-twin in training"
        )

    # Amendment 3 (2026-09-23, IEEE OJ-ITS submission review, B1): every
    # reported number used to come from `val_df`, while the manuscript stated
    # that metrics were computed on the test partition and never on
    # validation. `test_df` was split, leak-checked, and then never read.
    # Nothing was tuned or early-stopped on `val_df`, so the old numbers were
    # not optimistically biased — but the claim was false as written, and the
    # partition had been reused across months of iteration. All evaluation
    # now runs on `test_df`; `val_df` stays genuinely held out and is
    # deliberately unused, reserved for any future tuning. The parameter is
    # named `eval_df` downstream so this can never silently drift again.
    eval_df = test_df
    _log(f"[{name}] evaluation partition = TEST ({len(eval_df):,} rows); val held out, unused")

    _log(f"[{name}] Stage 1: training models {model_names or 'ALL'} on {len(train_df):,} rows")
    t0 = time.time()
    stage1 = train_stage1(
        train_df, eval_df, numeric_cols=numeric_cols, categorical_cols=CATEGORICAL_COLS,
        model_names=model_names, seed=seed,
    )
    _log(f"[{name}] Stage 1 done in {time.time() - t0:.1f}s")
    best_stage1_preds = stage1["models"]["lightgbm"]["eval_predictions"]
    stage1_metrics_by_model = _metrics_by_model(
        stage1["models"], lambda preds: family_metrics(stage1["family_true"], preds)
    )
    stage1_metrics = stage1_metrics_by_model["lightgbm"]
    _log(f"[{name}] Stage 1 macro_f1={stage1_metrics['macro_f1']:.4f} accuracy={stage1_metrics['accuracy']:.4f}")

    # Amendment 3 (B1/M4): a macro-F1 on a four-class problem whose majority
    # class holds 60% of the rows needs a floor to be readable. Stage 1's
    # accuracy on the original release sits below a constant predictor, which
    # a reader is entitled to see stated rather than having to derive it.
    stage1_trivial = trivial_baselines(
        train_df[FAMILY_LABEL_COL].values, stage1["family_true"], seed=seed
    )
    _log(
        f"[{name}] Stage 1 trivial floors: majority macro_f1="
        f"{stage1_trivial['majority_class']['macro_f1']:.4f} "
        f"(acc={stage1_trivial['majority_class']['accuracy']:.4f}), "
        f"stratified macro_f1={stage1_trivial['stratified_random']['macro_f1']:.4f}"
    )

    # Stage 1's own predicted routing (not the true family) — the subset a
    # real, deployed pipeline would actually send to Stage 2. May contain
    # false positives (true family is Normal/Classical) and miss false
    # negatives (a true PQ/Hybrid row Stage 1 routed elsewhere).
    routed_mask = np.isin(best_stage1_preds, ["Post-Quantum", "Hybrid"])
    eval_routed_df = eval_df[routed_mask].reset_index(drop=True)

    _log(f"[{name}] Stage 2: training models {model_names or 'ALL'} on PQ/Hybrid subset")
    t0 = time.time()
    stage2 = train_stage2(
        train_df, eval_df, numeric_cols=numeric_cols, categorical_cols=CATEGORICAL_COLS,
        model_names=model_names, predict_df=eval_routed_df, seed=seed,
    )
    _log(f"[{name}] Stage 2 done in {time.time() - t0:.1f}s")
    best_stage2_preds = stage2["models"]["lightgbm"]["eval_predictions"]
    stage2_metrics_by_model = _metrics_by_model(
        stage2["models"], lambda preds: subclass_metrics(stage2["subclass_true"], preds)
    )
    stage2_metrics = stage2_metrics_by_model["lightgbm"]
    _log(f"[{name}] Stage 2 (oracle-conditioned) macro_f1={stage2_metrics['macro_f1']:.4f}")

    train_pq_subtypes = train_df.loc[
        train_df[FAMILY_LABEL_COL].isin(["Post-Quantum", "Hybrid"]), "class_label"
    ].values
    stage2_trivial = trivial_baselines(train_pq_subtypes, stage2["subclass_true"], seed=seed)
    _log(
        f"[{name}] Stage 2 trivial floors: majority macro_f1="
        f"{stage2_trivial['majority_class']['macro_f1']:.4f}, "
        f"stratified macro_f1={stage2_trivial['stratified_random']['macro_f1']:.4f}"
    )

    routing_stats = routing_diagnostics(eval_df[FAMILY_LABEL_COL].values, best_stage1_preds)
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

    _log(f"[{name}] Flat baseline: training models {model_names or 'ALL'} on all {len(train_df):,} rows (24-class label)")
    t0 = time.time()
    flat = train_flat(
        train_df, eval_df, numeric_cols=numeric_cols, categorical_cols=CATEGORICAL_COLS,
        model_names=model_names, seed=seed,
    )
    _log(f"[{name}] Flat baseline done in {time.time() - t0:.1f}s")
    flat_eval_preds = flat["models"]["lightgbm"]["eval_predictions"]

    # Same rows Stage 1 sees (all of eval_df) and the same ground-truth family
    # column Stage 1 trains against — collapse the flat model's raw 24-class
    # predictions to family via the one true class->family map, so this is
    # directly comparable to stage1_metrics.macro_f1. The PQ/Hybrid subtype
    # comparison (comparable to stage2_metrics.macro_f1) is filtered by this
    # same TRUE family, not predicted family — see flat_comparison_metrics's
    # docstring for why that and the label-set restriction both matter.
    flat_family_true = eval_df[FAMILY_LABEL_COL].values
    flat_family_pred = derive_family_label(pd.Series(flat_eval_preds)).values

    flat_metrics = flat_comparison_metrics(
        flat["class_true"], flat_eval_preds, flat_family_true, flat_family_pred,
    )
    # Preserve the historical LightGBM comparison block above, and add the
    # same three comparisons for every other selected model.  This makes the
    # full-scale baseline run inspectable without changing old consumers.
    flat_metrics_by_model = {}
    for model_name, model_result in flat["models"].items():
        model_preds = model_result["eval_predictions"]
        model_family_preds = derive_family_label(pd.Series(model_preds)).values
        flat_metrics_by_model[model_name] = flat_comparison_metrics(
            flat["class_true"], model_preds, flat_family_true, model_family_preds,
        )
    flat_trivial = trivial_baselines(
        train_df["class_label"].values, flat["class_true"], seed=seed
    )
    _log(
        f"[{name}] Flat 24-class trivial floors: majority macro_f1="
        f"{flat_trivial['majority_class']['macro_f1']:.4f}, "
        f"stratified macro_f1={flat_trivial['stratified_random']['macro_f1']:.4f}"
    )
    _log(f"[{name}] Flat 24-class macro_f1={flat_metrics['flat_24class']['macro_f1']:.4f}")
    _log(f"[{name}] Flat family-derived macro_f1={flat_metrics['flat_family_derived']['macro_f1']:.4f}")
    _log(f"[{name}] Flat subtype(PQ/Hybrid subset) macro_f1={flat_metrics['flat_subtype_pq_hybrid']['macro_f1']:.4f}")

    # Symmetric comparison for Hypothesis 1: score the flat model on the
    # EXACT SAME rows Stage 1 actually routed to Stage 2 (not the
    # oracle-true-family subset flat_metrics["flat_subtype_pq_hybrid"]
    # uses) — same true-label-set restriction technique as
    # flat_comparison_metrics, for the same reason: the flat model can
    # predict labels outside routed_labels, Stage 2 cannot.
    flat_routed_pred = flat_eval_preds[routed_mask]
    flat_real_routing_metrics = subclass_metrics(
        stage2["routed_true"], flat_routed_pred, labels=routed_labels
    )
    _log(f"[{name}] Flat REAL-routing macro_f1={flat_real_routing_metrics['macro_f1']:.4f}")

    # Amendment 3 (M2): the two metrics above are computed only on rows Stage
    # 1 actually routed, so they charge the cascade for routing false
    # positives but stay silent about routing false negatives — a true
    # PQ/Hybrid attack Stage 1 sent elsewhere just leaves the evaluation set.
    # For a security detector that is the error that matters most. The
    # end-to-end view below keeps every truly PQ/Hybrid row in scope and
    # gives the never-routed ones a prediction that cannot be correct. The
    # flat classifier has no routing stage and never abstains, so its
    # end-to-end score is its score over the same rows, with an all-True mask.
    truly_pq_mask = np.isin(flat_family_true, ["Post-Quantum", "Hybrid"])
    true_pq_subtypes = flat["class_true"][truly_pq_mask]
    routed_within_true_pq = routed_mask[truly_pq_mask]
    stage2_preds_for_true_pq = stage2["routed_predictions"]["lightgbm"][truly_pq_mask[routed_mask]]

    stage2_end_to_end = end_to_end_subtype_metrics(
        true_pq_subtypes, routed_within_true_pq, stage2_preds_for_true_pq
    )
    flat_end_to_end = end_to_end_subtype_metrics(
        true_pq_subtypes,
        np.ones_like(routed_within_true_pq, dtype=bool),
        flat_eval_preds[truly_pq_mask],
    )
    _log(
        f"[{name}] END-TO-END subtype (all {stage2_end_to_end['n_total']:,} true PQ/Hybrid rows, "
        f"{stage2_end_to_end['n_never_routed']:,} never routed): "
        f"stage2={stage2_end_to_end['macro_f1']:.4f} flat={flat_end_to_end['macro_f1']:.4f}"
    )

    # Amendment 3 (M1): the headline claim is a ~0.02 macro-F1 gap. Give it
    # an interval. Both models are resampled on identical rows, so the shared
    # sampling noise cancels and what is left is their disagreement.
    bootstrap_metrics = {}
    if bootstrap:
        t0 = time.time()
        bootstrap_metrics["real_routing_flat_minus_stage2"] = paired_bootstrap_macro_f1(
            stage2["routed_true"], flat_routed_pred,
            stage2["routed_predictions"]["lightgbm"],
            labels=routed_labels, n_boot=n_boot, seed=seed,
        )
        bootstrap_metrics["end_to_end_flat_minus_stage2"] = paired_bootstrap_macro_f1(
            true_pq_subtypes,
            flat_eval_preds[truly_pq_mask],
            cascade_predictions(routed_within_true_pq, stage2_preds_for_true_pq),
            n_boot=n_boot, seed=seed,
        )
        for key, res in bootstrap_metrics.items():
            _log(
                f"[{name}] bootstrap {key}: observed={res['observed_difference']:+.4f} "
                f"95% CI [{res['ci95_lower']:+.4f}, {res['ci95_upper']:+.4f}] "
                f"P(flat>stage2)={res['fraction_a_greater']:.3f}"
            )
        _log(f"[{name}] bootstrap done in {time.time() - t0:.1f}s ({n_boot} resamples each)")

    real_cascade_metrics = {
        "routing_diagnostics": routing_stats,
        "stage2_real_routing": stage2_real_metrics,
        "flat_real_routing": flat_real_routing_metrics,
        "stage2_end_to_end": stage2_end_to_end,
        "flat_end_to_end": flat_end_to_end,
        "bootstrap": bootstrap_metrics,
        "stage2_real_routing_by_model": {
            name: subclass_metrics(
                stage2["routed_true"], stage2["routed_predictions"][name], labels=routed_labels
            )
            for name in stage2["models"]
        },
        "flat_real_routing_by_model": {
            name: subclass_metrics(
                stage2["routed_true"], result["eval_predictions"][routed_mask], labels=routed_labels
            )
            for name, result in flat["models"].items()
        },
    }

    real_synth_df = eval_df.iloc[: len(best_stage1_preds)].copy()
    real_synth_df["y_true"] = stage1["family_true"]
    real_synth_df["y_pred"] = best_stage1_preds
    real_vs_synth = compare_real_synthetic(real_synth_df, "y_true", "y_pred")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Amendment 3: every results file now records which partition produced it
    # and the trivial floors for its task, so a number can never again be
    # read without knowing where it came from. `convergence` (M5) lets a
    # reader tell a weak baseline from an under-trained one.
    provenance = {
        "evaluation_partition": "test",
        "n_eval_rows": int(len(eval_df)),
        "n_train_rows": int(len(train_df)),
        "n_val_rows_held_out_unused": int(len(val_df)),
        "seed": int(seed),
        "split": "grouped_by_feature_identity" if grouped_split else "plain_random",
        "test_rows_with_feature_twin_in_train": leakage_rate,
    }
    stage1_output = {
        **stage1_metrics,
        "by_model": stage1_metrics_by_model,
        "trivial_baselines": stage1_trivial,
        "convergence": _convergence_by_model(stage1["models"]),
        "provenance": provenance,
    }
    stage2_output = {
        **stage2_metrics,
        "by_model": stage2_metrics_by_model,
        "trivial_baselines": stage2_trivial,
        "convergence": _convergence_by_model(stage2["models"]),
        "provenance": provenance,
    }
    flat_output = {
        **flat_metrics,
        "by_model": flat_metrics_by_model,
        "trivial_baselines": flat_trivial,
        "convergence": _convergence_by_model(flat["models"]),
        "provenance": provenance,
    }
    real_cascade_metrics["provenance"] = provenance
    (RESULTS_DIR / f"{name}_stage1_metrics.json").write_text(json.dumps(_json_safe(stage1_output), indent=2))
    (RESULTS_DIR / f"{name}_stage2_metrics.json").write_text(json.dumps(_json_safe(stage2_output), indent=2))
    # Carries provenance like every other results file; the table generator
    # refuses to build from results that do not name their partition, and
    # this one was initially missed.
    real_vs_synth_output = {**real_vs_synth, "provenance": provenance}
    (RESULTS_DIR / f"{name}_real_vs_synthetic.json").write_text(
        json.dumps(_json_safe(real_vs_synth_output), indent=2)
    )
    _log(f"[{name}] wrote results/{name}_stage1_metrics.json, {name}_stage2_metrics.json, {name}_real_vs_synthetic.json")

    (RESULTS_DIR / f"{name}_flat_metrics.json").write_text(json.dumps(_json_safe(flat_output), indent=2))
    _log(f"[{name}] wrote results/{name}_flat_metrics.json")

    (RESULTS_DIR / f"{name}_real_cascade_metrics.json").write_text(
        json.dumps(_json_safe(real_cascade_metrics), indent=2)
    )
    _log(f"[{name}] wrote results/{name}_real_cascade_metrics.json")

    return {
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "flat": flat_metrics,
        "real_cascade": real_cascade_metrics,
        "provenance": provenance,
        "trivial_baselines": {
            "stage1": stage1_trivial,
            "stage2": stage2_trivial,
            "flat": flat_trivial,
        },
        "convergence": {
            "stage1": _convergence_by_model(stage1["models"]),
            "stage2": _convergence_by_model(stage2["models"]),
            "flat": _convergence_by_model(flat["models"]),
        },
        "by_model": {
            "stage1": stage1_metrics_by_model,
            "stage2": stage2_metrics_by_model,
            "flat": flat_metrics_by_model,
        },
    }


if __name__ == "__main__":
    pipeline_start = time.time()

    # Operational addition (not in the brief's literal listing): Fix 1/Fix 2
    # only changed the in-memory feature/split logic in run_corpus, not the
    # raw-to-Parquet conversion (csv_to_parquet/h5_to_parquet), so a Parquet
    # cache from a previous real run is still valid and safe to reuse — and
    # worth reusing, since the H5 conversion reads its source file from a
    # Google-Drive-hosted path and previously took a large fraction of the
    # multi-hour run. Re-convert only if the cache is missing.
    csv_parquet = CSV_CORPUS_DIR / "data.parquet"
    if csv_parquet.exists():
        _log(f"Reusing existing CSV Parquet cache -> {csv_parquet}")
    else:
        _log("Converting raw CSV to Parquet")
        t0 = time.time()
        csv_parquet = csv_to_parquet(RAW_CSV_PATH, CSV_CORPUS_DIR)
        _log(f"CSV -> Parquet done in {time.time() - t0:.1f}s -> {csv_parquet}")

    h5_parquet = H5_CORPUS_DIR / "data.parquet"
    if h5_parquet.exists():
        _log(f"Reusing existing H5 Parquet cache -> {h5_parquet}")
    else:
        _log("Converting raw H5 to Parquet")
        t0 = time.time()
        h5_parquet = h5_to_parquet(RAW_H5_PATH, H5_CORPUS_DIR)
        _log(f"H5 -> Parquet done in {time.time() - t0:.1f}s -> {h5_parquet}")

    _log("=== Running CSV corpus ===")
    csv_results = run_corpus("csv", csv_parquet, extra_numeric_cols=[], model_names=REAL_RUN_MODEL_NAMES)

    _log("=== Running H5 corpus ===")
    h5_results = run_corpus(
        "h5", h5_parquet, extra_numeric_cols=H5_EXTRA_NUMERIC_COLS, model_names=REAL_RUN_MODEL_NAMES
    )

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
        "csv_stage2_real_routing_macro_f1": csv_results["real_cascade"]["stage2_real_routing"]["macro_f1"],
        "h5_stage2_real_routing_macro_f1": h5_results["real_cascade"]["stage2_real_routing"]["macro_f1"],
        "csv_flat_real_routing_macro_f1": csv_results["real_cascade"]["flat_real_routing"]["macro_f1"],
        "h5_flat_real_routing_macro_f1": h5_results["real_cascade"]["flat_real_routing"]["macro_f1"],
    }
    (RESULTS_DIR / "csv_vs_h5_comparison.json").write_text(json.dumps(_json_safe(comparison), indent=2))
    _log(f"Total pipeline time: {time.time() - pipeline_start:.1f}s")
    print(json.dumps(_json_safe(comparison), indent=2))
