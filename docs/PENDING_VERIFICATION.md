# Claims written into the manuscript that are NOT yet verified

Last checked 2026-09-25 against the completed test-partition run.

- [x] "direction of the reversal depends on the scored row set" — DISPROVEN
      on the real CSV corpus (flat wins under both deployable scopes).
      Sentence corrected on 2026-09-23. Confirmed on the H5 corpus too; the
      end-to-end comparison is now reported as inconclusive for the extended
      release, because seed 123 reverses it.
- [x] Every number in the tables — regenerating from `results/` on 2026-09-25
      reproduced every committed table byte for byte.
- [x] "This reversal persists across all three seeds" (abstract) — holds under
      predicted routing for all three seeds in both releases, checked seed by
      seed in `results/multiseed/summary.json`.
- [x] Abstract numbers — now the test-partition values.
- [x] Runtime table — replaced with the wall-clock of
      `logs/full_run_test_partition_20260923_110535.log`
      (split 179.3 s / 89.3 s; Stage 1 63.4 / 105.0 min; Stage 2 31.1 / 41.4
      min; flat 77.6 / 130.7 min).
- [ ] The four per-model LightGBM training times in the Implementation section
      (287.5 / 219.3 / 230.9 / 243.0 s) are from the superseded run. The new
      pipeline records wall-clock per stage for the four models together, not
      per model, so these cannot be verified from current artifacts. Decide:
      drop the sentence (the runtime table already reports cost) or re-measure
      LightGBM alone.
- [ ] Hand-typed tables not derived from `results/`: family distribution,
      leakage audit AUCs, hyperparameters. They are dataset-level or
      configuration facts, unaffected by the partition correction, but no
      script regenerates them, so they rest on the 2026-08 EDA.
