# Kernel-Based Analysis of Preference Functions

This repository contains the anonymized empirical data and the Python code used
for the simulations and empirical case study reported in *A Kernel-Based
Framework for Analyzing Individual Differences in Preference Functions*.

Version `0.2.0` provides the data, code, and results corresponding to the
pre-submission manuscript revised on September 9, 2026. The associated article
has not yet been submitted or accepted. A Zenodo archive and DOI may be added
when the manuscript and reproducibility materials are finalized.

## Repository structure

- `empirical/`: anonymized ratings from 20 participants, x-block cross-validation,
  KRR--RKHS-PCA, clustering, numerical results, and figure-generation code.
- `simulation/`: the two Monte Carlo studies, a supplementary `T=20, N=50`
  sensitivity analysis, saved numerical results, and manuscript figures.
- `requirements.txt`: Python dependencies shared by both analyses.

Earlier exploratory analyses, pilot runs, manuscript drafts, copyrighted
reference PDFs, and obsolete code are intentionally excluded.

## Environment

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Empirical analysis

To reproduce the complete 231-condition x-block cross-validation and refit the
model:

```bash
python empirical/code/run_analysis.py
python empirical/code/make_final_figures.py
```

To skip cross-validation and refit the model with the saved selected parameters:

```bash
python empirical/code/fit_selected_model.py
python empirical/code/make_final_figures.py
```

The selected values are `length = 0.3162277660` (reported as `0.32`) and
`beta = 0.0031697864` (reported as `0.0032`) in the manuscript's
mean-squared-loss parameterization. Within each fold and participant, the mean
and population SD (`ddof=0`) are computed from the training ratings and then
applied to both training and held-out ratings. The 231 combinations are
the Cartesian product of 21 logarithmically spaced length scales from `0.1` to
`1.0` and 11 logarithmically spaced `beta` values from `0.0002` to `0.02`.

Cross-validation, final fitting, and sensitivity analysis use
`alpha_t = solve(K_tt + N_t * beta * I, s_t)` without additional jitter.
RKHS inner products are computed directly from participant-specific
coefficients and cross-kernel matrices, following the manuscript's Method
section and the same equations used in the simulations. Regenerate the
unversioned fitted-model cache with the current code before plotting.

A historical standardization sensitivity analysis is retained as a working
record, separately described in `empirical/README.md`. It is not part of the
current manuscript and is not required to reproduce the primary analysis.

## Simulation 1: pointwise shape reconstruction

```bash
python simulation/code/run_metric_ablation.py
```

This runs 200 Monte Carlo replications comparing Raw-PCA, KRR--L2-PCA, and
KRR--RKHS-PCA. The two KRR pipelines use the same fitted functions and differ
only in the PCA metric.

## Simulation 2: latent-subspace recovery

Main sensitivity analysis (`T=30`, `N` in `{15, 30, 50}`, and SNR in
`{0.5, 1, 2}`):

```bash
python simulation/code/run_latent_subspace_sensitivity.py
```

Supplementary empirical-design sensitivity analysis (`T=20`, `N=50`; briefly
reported in the main manuscript):

```bash
python simulation/code/run_latent_subspace_sensitivity_T20_N50.py
```

To regenerate the simulation figures from the saved numerical summaries:

```bash
python simulation/code/make_manuscript_simulation_figures.py
```

All Monte Carlo scripts use fixed random seeds. Full simulations and the
empirical cross-validation may take substantial time; the saved CSV files in
the corresponding `results/` directories are the outputs used in the
manuscript.

## Validate the release package

```bash
python validate_release.py
python empirical/code/validate_model.py
```

This checks the empirical data structure, aspect-ratio transformation, selected
parameters, RKHS variance proportions, cluster sizes, and the numerical values
reported from both simulations. The model checks compare the dual calculation
with an independent feature-space solution and verify PCA invariants.

## Data and privacy

The empirical files contain stimulus aspect ratios and ordinal beauty ratings.
Participant numbers are arbitrary study identifiers; names, contact details,
dates, and other direct identifiers are not included. See
`empirical/DATA_DICTIONARY.md` for the variable definitions.

## Citation and licenses

Citation metadata are provided in `CITATION.cff`. The Python source code is
licensed under the MIT License in `LICENSE`. The data,
numerical results, generated figures, and documentation are licensed under
CC BY 4.0 as specified in `LICENSE-DATA.md`. No DOI has been assigned to this
release; use the repository URL and version in `CITATION.cff` for now.
