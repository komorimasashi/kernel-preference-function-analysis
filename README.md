# Kernel-Based Analysis of Preference Functions

This repository contains the anonymized empirical data and the Python code used
for the simulations and empirical case study reported in *Modeling the
Functions of Preferences: A Kernel-Based Framework for Analyzing Individual
Differences*.

The repository is currently being prepared as a private working copy. A
versioned public release will be archived on Zenodo when the manuscript and
reproducibility materials are finalized.

## Repository structure

- `empirical/`: anonymized ratings from 20 participants, x-block cross-validation,
  KRR--RKHS-PCA, clustering, numerical results, and figure-generation code.
- `simulation/`: the two Monte Carlo studies, the additional `T=20, N=50`
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

The selected values are `length = 0.4466835922` (reported as `0.45`) and
`beta = 0.02` in the manuscript's mean-squared-loss parameterization.

## Simulation 1: pointwise shape reconstruction

```bash
python simulation/code/run_metric_ablation_expanded_grid.py
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

Additional empirical-design analysis (`T=20`, `N=50`):

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
```

This checks the empirical data structure, aspect-ratio transformation, selected
parameters, RKHS variance proportions, cluster sizes, and the numerical values
reported from both simulations.

## Data and privacy

The empirical files contain stimulus aspect ratios and ordinal beauty ratings.
Participant numbers are arbitrary study identifiers; names, contact details,
dates, and other direct identifiers are not included. See
`empirical/DATA_DICTIONARY.md` for the variable definitions.

## Citation and licenses

Citation metadata are provided in `CITATION.cff`. The Python source code and
Jupyter notebook are licensed under the MIT License in `LICENSE`. The data,
numerical results, generated figures, and documentation are licensed under
CC BY 4.0 as specified in `LICENSE-DATA.md`. The Zenodo DOI will be added to all
three files before the public release.
