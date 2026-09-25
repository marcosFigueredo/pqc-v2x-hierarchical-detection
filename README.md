# PQ-V2X hierarchical versus flat intrusion detection

Computational record for the study *Reversal of Hierarchical Advantage in V2X
Intrusion Detection for Post-Quantum Intelligent Transportation Systems*.

The study asks whether a two-stage detector that first identifies the attack
family and then the subtype beats a single flat 24-class classifier, on the two
five-million-record releases of the PQ-V2X dataset. It answers under a
leakage-aware protocol, and the answer depends on who supplies the routing.

## The finding

Neither PQ-V2X release holds five million independent observations, so every
partition here is grouped by predictor-value identity, and every reported
metric comes from a held-out test partition that the pipeline never fits or
tunes on. Under a plain random split of the same data, the same model reaches
macro-F1 above 0.98, which measures what duplicate-pattern leakage contributes
rather than what the model learned.

Macro-F1 with LightGBM on the test partition:

| Level and routing | Original release | Extended release |
| --- | --- | --- |
| Family, Stage 1 versus flat | **0.5864** vs 0.4962 | **0.8796** vs 0.8726 |
| Subtype, oracle routing, Stage 2 versus flat | **0.4233** vs 0.3757 | **0.4289** vs 0.4130 |
| Subtype, predicted routing, cascade versus flat | 0.3236 vs **0.3479** | 0.3753 vs **0.3903** |

The hierarchy wins at the family level, and wins at the subtype level as long
as an oracle hands Stage 2 the true family. Once Stage 1 supplies the routing,
the ordering reverses and the flat classifier wins. That reversal holds in both
releases, in all three seeds (42, 123, 2024), and its paired bootstrap interval
excludes zero. An oracle-only evaluation therefore reports the opposite
architectural conclusion from a deployable cascade.

## What is in here

| Path | Contents |
| --- | --- |
| `src/` | The pipeline: splitting, feature selection, Stage 1, Stage 2, flat baseline, metrics |
| `tests/` | 42 pytest checks, including a guard that fails if evaluation leaves the test partition |
| `scripts/` | Multi-seed run, split ablation, table and figure generation, Zenodo archive builder |
| `results/` | Every result file behind every table, each carrying a `provenance` block naming its partition |
| `logs/` | Wall-clock run logs of the runs that produced those results |
| `docs/` | Design spec and implementation plans, plus the open verification items |
| `ojits-latex-template/` | Manuscript source; `generated/` holds the tables and figure built from `results/` |

Every numeric table and the confusion-matrix figure in the manuscript is
generated from `results/` by `scripts/make_figures_and_tables.py` and included
with `input`, so a re-run cannot silently leave a stale number in the paper.

## Reproducing

See [REPRODUCING.md](REPRODUCING.md) for the environment, the pinned package
versions, and the order of the runs. In short:

```
pip install -r requirements.txt
python -m pytest tests/ -q
python -m src.run_pipeline
```

## Data

The PQ-V2X dataset is not redistributed here. It was created and is
distributed by its own authors (doi:10.1109/JIOT.2025.3618153). Place the two
raw releases where `src/config.py` expects them, or edit those paths.
[PQ-V2X_Original_Paper_Benchmark_README.md](PQ-V2X_Original_Paper_Benchmark_README.md)
records the exact release this study treats as the original benchmark,
including its SHA-256.

## Citing

The archived version of this repository, with the result files, carries its own
DOI: [10.5281/zenodo.22965040](https://doi.org/10.5281/zenodo.22965040). That
concept DOI always resolves to the most recent archived version. Please cite
the paper as well once it appears.

## License

The code is MIT licensed; see [LICENSE](LICENSE). The contents of
`ojits-latex-template/` are the manuscript, not code: they remain the authors'
work and will fall under the publisher's copyright terms if the paper is
accepted.
