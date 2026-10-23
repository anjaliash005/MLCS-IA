# IA-2 — Security and Privacy for Machine Learning

**Anjali Ashtankar · 16014223012 · Batch B1**

*Membership Inference as a Common Yardstick: Evaluating Five Families of
Privacy-Preserving Defences for Healthcare Machine Learning*

The IA-1 membership inference attack (shadow models on a Pima diabetes classifier)
is used as the shared privacy metric for one defence family from every module:

| Module | What is implemented | File |
|---|---|---|
| M1 Privacy in ML | Shadow-model + loss-threshold MIA, label-only/rounded APIs, model matrix A–E | `src/attacks.py`, `src/models.py` |
| M2 Differential privacy | DP gradient descent + zCDP accountant, DP advantage bound, DP k-means | `src/m2_dp.py` |
| M3 Local DP | Laplace / Duchi / Piecewise / Hybrid, randomised response, OUE, LDP Naive Bayes | `src/m3_ldp.py` |
| M4 PPDM | Mondrian k-anonymity, l-diversity, t-closeness, input noise, output restriction | `src/m4_ppdm.py` |
| M5 Compressive privacy | PCA, DCA, DUCA projections; utility vs privacy task, reconstruction | `src/m5_compressive.py` |

## Deliverables

- `reports/IA1_Report_Anjali_Ashtankar_16014223012.pdf` — IA-1 report (problem formulation, five-module
  mapping, dataset preparation, baseline attack on seed 0, mechanism prototypes); rebuilt with
  `src/run_ia1_baseline.py` then `src/build_ia1_report.py`
- `reports/IA2_Technical_Report_Anjali_Ashtankar_16014223012.pdf` — final technical report (IA-2 stages 1–5)
- `reports/IA2_Manuscript_Draft_Anjali_Ashtankar_16014223012.pdf` — IEEE-style manuscript draft
- `results/tables/` — per-seed results, summaries, paired statistics, provenance (`provenance.json`)
- `results/figures/` — all figures
- `data/raw/` (hash-verified source), `data/processed/` (prepared datasets)

## Reproduce

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.lock.txt
```

```bash
.venv/bin/python src/run_experiments.py
```

```bash
.venv/bin/python src/make_figures.py && .venv/bin/python src/build_report.py && .venv/bin/python src/build_manuscript.py
```

`run_experiments.py --quick` runs a 2-seed smoke test. The full run (10 seeds, 8 worker
processes) took about 16 minutes on a 10-core laptop.

## Data

- **Pima Indians Diabetes** — `data/raw/pima-indians-diabetes.data.csv`, downloaded from the
  jbrownlee/Datasets GitHub mirror of the UCI file, SHA-256
  `6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af`. The mirror's licence is not stated.
- **Breast Cancer Wisconsin (Diagnostic)** — copy bundled with scikit-learn 1.6.1 (UCI, CC BY 4.0).

Both are public and de-identified; experiments only attack locally trained models.
