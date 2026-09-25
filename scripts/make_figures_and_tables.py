"""Generate every numeric LaTeX table and figure from results/*.json.

The manuscript carried roughly forty hand-typed numbers across its tables and
a hand-built confusion-matrix figure. Any re-run silently invalidated all of
them, and a reader auditing the archive against the PDF would find the
mismatch before the authors did. Everything numeric is emitted here instead,
so the PDF cannot drift from the results it reports.

Writes .tex fragments into `ojits-latex-template/generated/`, which the
manuscript \\input{}s. Do not hand-edit the generated files.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "ojits-latex-template" / "generated"

RELEASES = [("csv", "Orig."), ("h5", "Ext.")]
MODEL_ROWS = [
    ("lightgbm", "LightGBM"),
    ("logreg", "Logistic regression"),
    ("random_forest", "Random forest"),
    ("mlp", "MLP"),
]
FAMILY_SHORT = {
    "Classical": "Cla",
    "Hybrid": "Hyb",
    "Normal": "Nor",
    "Post-Quantum": "PQ",
}


def load(name):
    path = RESULTS / name
    if not path.exists():
        sys.exit(f"missing {path}; run the pipeline first")
    data = json.loads(path.read_text())
    partition = data.get("provenance", {}).get("evaluation_partition")
    if partition != "test":
        sys.exit(
            f"{path.name} reports evaluation_partition={partition!r}, not 'test'. "
            "These results predate the test-partition correction; re-run "
            "`python -m src.run_pipeline` before generating tables."
        )
    return data


def fmt(x, places=4):
    return f"{x:.{places}f}"


def bold(value, places=4):
    return r"\textbf{" + fmt(value, places) + "}"


def bold_best(values, places=4):
    best = max(values)
    return [bold(v, places) if v == best else fmt(v, places) for v in values]


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body, encoding="utf8")
    print(f"wrote generated/{name}")


def table_stage_by_model():
    s1 = {r: load(f"{r}_stage1_metrics.json") for r, _ in RELEASES}
    s2 = {r: load(f"{r}_stage2_metrics.json") for r, _ in RELEASES}

    columns = [
        [s1[r]["by_model"][m]["macro_f1"] for m, _ in MODEL_ROWS] for r, _ in RELEASES
    ] + [
        [s2[r]["by_model"][m]["macro_f1"] for m, _ in MODEL_ROWS] for r, _ in RELEASES
    ]
    formatted = [bold_best(col) for col in columns]

    rows = []
    for i, (_, label) in enumerate(MODEL_ROWS):
        cells = " & ".join(col[i] for col in formatted)
        rows.append(f"{label} & {cells} \\\\")

    def floor_row(kind, label):
        cells = " & ".join(
            fmt(source[r]["trivial_baselines"][kind]["macro_f1"])
            for source in (s1, s2)
            for r, _ in RELEASES
        )
        return f"{label} & {cells} \\\\"

    header = r"""\begin{table}[!t]
\caption{Macro-F1 by model on the test partition, for attack-family
detection (Stage 1) and exact-subtype detection under oracle routing
(Stage 2).
Orig.\ = original release, Ext.\ = extended release. Best model per column in
bold; the last two rows are trivial predictors, not trained models}
\label{tab:stage_by_model}
\centering
\begin{tabular}{lrrrr}
\toprule
 & \multicolumn{2}{c}{Stage 1 (family)} & \multicolumn{2}{c}{Stage 2 (subtype, oracle)} \\
Model & Orig. & Ext. & Orig. & Ext. \\
\midrule
"""
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    body = (
        header
        + "\n".join(rows)
        + "\n\\midrule\n"
        + floor_row("majority_class", "Majority class")
        + "\n"
        + floor_row("stratified_random", "Stratified random")
        + footer
    )
    write("tab_stage_by_model.tex", body)


def table_flat_vs_hier():
    s1 = {r: load(f"{r}_stage1_metrics.json") for r, _ in RELEASES}
    s2 = {r: load(f"{r}_stage2_metrics.json") for r, _ in RELEASES}
    flat = {r: load(f"{r}_flat_metrics.json") for r, _ in RELEASES}
    casc = {r: load(f"{r}_real_cascade_metrics.json") for r, _ in RELEASES}

    blocks = [
        (
            r"Family ($\mathcal{F}$)", r"$f_1$ (Stage 1)", "",
            (s1["csv"]["macro_f1"], s1["h5"]["macro_f1"]),
            (flat["csv"]["flat_family_derived"]["macro_f1"],
             flat["h5"]["flat_family_derived"]["macro_f1"]),
        ),
        (
            r"Subtype ($\mathcal{Y}_q$)", r"$f_2$ (Stage 2)", "Oracle",
            (s2["csv"]["macro_f1"], s2["h5"]["macro_f1"]),
            (flat["csv"]["flat_subtype_pq_hybrid"]["macro_f1"],
             flat["h5"]["flat_subtype_pq_hybrid"]["macro_f1"]),
        ),
        (
            r"Subtype ($\mathcal{Y}_q$)", r"$f_2$ (cascade)", "Predicted",
            (casc["csv"]["stage2_real_routing"]["macro_f1"],
             casc["h5"]["stage2_real_routing"]["macro_f1"]),
            (casc["csv"]["flat_real_routing"]["macro_f1"],
             casc["h5"]["flat_real_routing"]["macro_f1"]),
        ),
        (
            r"Subtype ($\mathcal{Y}_q$)", r"$f_2$ (cascade)", "End-to-end",
            (casc["csv"]["stage2_end_to_end"]["macro_f1"],
             casc["h5"]["stage2_end_to_end"]["macro_f1"]),
            (casc["csv"]["flat_end_to_end"]["macro_f1"],
             casc["h5"]["flat_end_to_end"]["macro_f1"]),
        ),
    ]

    lines = []
    for level, hier_name, condition, hier, flat_values in blocks:
        hier_cells = [
            bold(h) if h > f else fmt(h) for h, f in zip(hier, flat_values)
        ]
        flat_cells = [
            bold(f) if f > h else fmt(f) for h, f in zip(hier, flat_values)
        ]
        lines.append(
            f"{level} & {hier_name} & {condition} & {hier_cells[0]} & {hier_cells[1]} \\\\"
        )
        lines.append(
            f"{level} & $f_{{\\mathrm{{flat}}}}$ & {condition} & "
            f"{flat_cells[0]} & {flat_cells[1]} \\\\"
        )
        lines.append(r"\midrule")
    lines.pop()

    body = r"""\begin{table}[!t]
\caption{Flat baseline versus hierarchical architecture on the test partition
(LightGBM, macro-F1). Oracle supplies the true family; Predicted scores only
the rows Stage 1 routed; End-to-end scores every truly PQ/Hybrid row, counting
those the cascade never routed as errors. Better value of each pair in bold}
\label{tab:flat_vs_hier}
\centering
\small
\begin{tabular}{lllrr}
\toprule
Level & Model & Condition & Original & Extended \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_flat_vs_hier.tex", body)


def table_bootstrap():
    casc = {r: load(f"{r}_real_cascade_metrics.json") for r, _ in RELEASES}
    labels = [
        ("real_routing_flat_minus_stage2", "Predicted routing"),
        ("end_to_end_flat_minus_stage2", "End-to-end"),
    ]
    rows = []
    n_boot = None
    for key, label in labels:
        cells = []
        for r, _ in RELEASES:
            entry = casc[r].get("bootstrap", {}).get(key)
            if entry is None:
                cells.append("n/a")
                continue
            n_boot = entry["n_boot"]
            cells.append(
                f"${entry['observed_difference']:+.4f}$ "
                f"$[{entry['ci95_lower']:+.4f}, {entry['ci95_upper']:+.4f}]$"
            )
        rows.append(f"{label} & {cells[0]} & {cells[1]} \\\\")

    count = f"{n_boot:,}" if n_boot else "1{,}000"
    body = r"""\begin{table*}[!t]
\caption{Paired bootstrap of the flat-minus-cascade macro-F1 difference
(LightGBM, """ + count + r""" resamples of the same rows, 95\% percentile
interval). A positive value favors the flat classifier; an interval excluding
zero separates the two architectures}
\label{tab:bootstrap}
\centering
\small
\begin{tabular}{lcc}
\toprule
Condition & Original & Extended \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_bootstrap.tex", body)


def table_multiseed():
    path = RESULTS / "multiseed" / "summary.json"
    if not path.exists():
        print("note: multiseed summary absent, skipping tab_multiseed")
        return
    data = json.loads(path.read_text())
    seeds = sorted((s for s in data if s.isdigit()), key=int)

    metrics = [
        ("stage1_macro_f1", "Stage 1"),
        ("stage2_oracle_macro_f1", "Stage 2, oracle"),
        ("stage2_real_routing_macro_f1", "Stage 2, predicted routing"),
        ("flat_real_routing_macro_f1", "Flat, predicted routing"),
        ("stage2_end_to_end_macro_f1", "Stage 2, end-to-end"),
        ("flat_end_to_end_macro_f1", "Flat, end-to-end"),
        ("routing_precision", "Routing precision"),
        ("routing_recall", "Routing recall"),
    ]
    rows = []
    for key, label in metrics:
        cells = []
        for release, _ in RELEASES:
            values = [data[s][release].get(key) for s in seeds]
            if any(v is None for v in values):
                cells.append("n/a")
                continue
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            cells.append(f"{mean:.4f} $\\pm$ {std:.4f}")
        if all(c == "n/a" for c in cells):
            continue
        rows.append(f"{label} & {cells[0]} & {cells[1]} \\\\")

    body = r"""\begin{table}[!t]
\caption{Multi-seed robustness on the test partition (LightGBM, mean $\pm$
population standard deviation across seeds """ + ", ".join(seeds) + r"""; macro-F1
unless noted)}
\label{tab:multiseed}
\centering
\small
\begin{tabular}{lrr}
\toprule
Metric & Original & Extended \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_multiseed.tex", body)


def table_real_synthetic():
    data = {r: load(f"{r}_real_vs_synthetic.json") for r, _ in RELEASES}
    rows = []
    for release, label in (("csv", "Original"), ("h5", "Extended")):
        real = data[release]["real"]
        synthetic = data[release]["synthetic"]
        f1s = bold_best([real["macro_f1"], synthetic["macro_f1"]])
        accuracies = bold_best([real["accuracy"], synthetic["accuracy"]])
        rows.append(f"{label} & Real & {f1s[0]} & {accuracies[0]} \\\\")
        rows.append(f"{label} & Synthetic & {f1s[1]} & {accuracies[1]} \\\\")

    body = r"""\begin{table}[!t]
\caption{Stage 1 on the test partition (LightGBM, family level), real
versus synthetic traces. Higher value of each pair in bold}
\label{tab:real_synthetic}
\centering
\begin{tabular}{llrr}
\toprule
Release & Trace & Macro-F1 & Accuracy \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_real_synthetic.tex", body)


def table_split_ablation():
    path = RESULTS / "split_ablation" / "summary.json"
    if not path.exists():
        print("note: split ablation absent, skipping tab_split_ablation")
        return
    data = json.loads(path.read_text())
    rows = []
    for release, label in (("csv", "Original"), ("h5", "Extended")):
        for condition, condition_label in (
            ("grouped", "Grouped by feature identity"),
            ("random", "Plain random"),
        ):
            entry = data[condition][release]
            twinned = 100.0 * entry["test_rows_with_feature_twin_in_train"]
            rows.append(
                f"{label} & {condition_label} & {twinned:.1f}\\% & "
                f"{fmt(entry['stage1_macro_f1'])} & "
                f"{fmt(entry['flat_24class_macro_f1'])} \\\\"
            )

    body = r"""\begin{table*}[!t]
\caption{Split ablation (LightGBM, identical data, model, and scale; only the
splitting rule changes). Twinned is the share of test rows holding an exact
predictor-value twin in the training partition}
\label{tab:split_ablation}
\centering
\small
\begin{tabular}{llrrr}
\toprule
Release & Split & Twinned & Stage 1 & Flat 24-class \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_split_ablation.tex", body)


def figure_confusion():
    data = {r: load(f"{r}_stage1_metrics.json") for r, _ in RELEASES}
    parts = []
    panels = (
        ("csv", "Original release", 1.35, 0.0),
        ("h5", "Extended release", -2.5, -3.85),
    )
    for release, title, title_y, offset in panels:
        entry = data[release]
        labels = entry["labels"]
        matrix = entry["confusion_matrix"]
        parts.append(
            f"\\node[font=\\small\\bfseries] at (2.45,{title_y}) {{{title}}};"
        )
        for j, label in enumerate(labels):
            parts.append(
                f"\\node[lbl] at ({1.4 + j:.1f},{0.9 + offset:.2f}) "
                f"{{{FAMILY_SHORT[label]}}};"
            )
        for i, label in enumerate(labels):
            y = 0.34 + offset - 0.68 * i
            parts.append(
                f"\\node[lbl] at (0.0,{y:.2f}) {{{FAMILY_SHORT[label]}}};"
            )
            row_total = sum(matrix[i]) or 1
            for j in range(len(labels)):
                percentage = 100.0 * matrix[i][j] / row_total
                shade = int(round(percentage))
                text = ", text=white" if percentage >= 50 else ""
                parts.append(
                    f"\\node[cell, fill=blue!{shade}{text}] at "
                    f"({1.4 + j:.1f},{y:.2f}) {{{percentage:.1f}}};"
                )

    body = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}[
  every node/.style={font=\tiny},
  cell/.style={draw=gray!50, minimum width=1.0cm, minimum height=0.68cm, align=center},
  lbl/.style={font=\scriptsize},
]
""" + "\n".join(parts) + r"""
\end{tikzpicture}
\caption{Stage 1 confusion matrices on the test partition (LightGBM,
row-normalized \%). Rows are the true family, columns the predicted family.
Cla = Classical, Hyb = Hybrid, Nor = Normal, PQ = Post-Quantum.}
\label{fig:confusion_stage1}
\end{figure}
"""
    write("fig_confusion_stage1.tex", body)


def main():
    table_stage_by_model()
    table_flat_vs_hier()
    table_bootstrap()
    table_multiseed()
    table_real_synthetic()
    table_split_ablation()
    figure_confusion()
    print(
        "\nAll numeric floats regenerated from results/. "
        "The manuscript inputs these; do not hand-edit them."
    )


if __name__ == "__main__":
    main()
