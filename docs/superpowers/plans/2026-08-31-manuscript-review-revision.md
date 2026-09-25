# Manuscript Review Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Revise the PQ-V2X manuscript for IEEE TVT readiness by translating the body to academic English and addressing the reviewer's methodological and bibliographic concerns.

**Architecture:** Keep the current experimental design and results unchanged unless a verified data audit requires a new split. Add a concise threat model, clarify the scope of every metric, explain the extended-release separation, and preserve the distinction between the scikit-learn package and the models it provides.

**Tech Stack:** LaTeX, IEEEtran, BibTeX, Python, pandas, scikit-learn, LightGBM.

**Spec:** Reviewer report supplied by the user in the current session.

## Global Constraints

- The manuscript must remain at or below 13 pages.
- The body text must be in English; tables, figures, captions, and technical labels remain in English.
- Do not describe scikit-learn as a model. Describe it as a Python package and name each model separately.
- Preserve manual edits before every modification and use only verified results.
- Avoid overstated competitive language and distinguish oracle routing from predicted routing.

### Task 1  Audit data identifiers and current manuscript claims

**Files:** `src/run_pipeline.py`, `src/data_prep.py`, `main.tex`, experiment logs.

- [ ] Verify whether `session_id` and `vehicle_id` are available and non-degenerate in each corpus.
- [ ] Record the actual grouping rule used by the reproducible pipeline.
- [ ] Identify all Portuguese prose, the routing-error interpretation, the 0.382 versus 0.3800 discrepancy, and missing software citations.

### Task 2  Add threat model and methodological clarifications

**Files:** `main.tex`.

- [ ] Add a concise English threat-model paragraph defining external and internal attackers, passive observation, active injection, replay, spoofing, and manipulation of PQC-related traffic.
- [ ] State that the detector observes communication, timing, channel, mobility, network, and cryptographic-derived attributes but does not assume compromise of the classifier or labels at deployment.
- [ ] Explain that the experimental target is attack-family and subtype recognition, not cryptographic protocol validation.
- [ ] Correct the confusion-matrix interpretation and explicitly discuss the suspicious perfect Normal/PQ separation in the extended release as a dataset-design signal requiring cautious interpretation.

### Task 3  Correct tables and references

**Files:** `main.tex`, `references.bib`.

- [ ] Explain the metric scope difference between the 24-class flat result and the PQC-subtype comparison, or align the values if they refer to the same test set.
- [ ] Add LightGBM and scikit-learn references, describing scikit-learn as a package.
- [ ] Preserve normalized DOI URLs and complete missing DOI links.

### Task 4  Translate the manuscript body

**Files:** `main.tex`.

- [ ] Translate Portuguese prose in Sections I through IX into concise academic English.
- [ ] Preserve equations, labels, citations, tables, figures, captions, and technical variable names.
- [ ] Maintain neutral interpretation of numerical comparisons and keep the objective aligned with the research questions.

### Task 5  Verify the revised manuscript

**Files:** generated `main.pdf`, `main.log`, `main.bbl`.

- [ ] Run BibTeX and two LaTeX passes.
- [ ] Check for LaTeX errors, undefined references, missing authors, proxy DOI domains, and page count.
- [ ] Review the diff and confirm no unrelated content changed.
