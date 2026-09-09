# Empirical Case Study

This directory contains the anonymized ratings and the final empirical
KRR--RKHS-PCA analysis reported in the manuscript.

## Main results

- Selected RBF length scale: `0.3162277660` (reported as `0.32`).
- Selected KRR regularization in the manuscript parameterization:
  `beta = 0.0031697864` (reported as `0.0032`).
- Mean validation RMSE after training-fold standardization: `0.9206`.
- RKHS variance proportions: PC1 `45.31%`, PC2 `32.81%`, PC3 `9.88%`.
- Cumulative RKHS variance: first two PCs `78.12%`; first three PCs `88.00%`.
- Descriptive four-group partition: cluster sizes `5`, `4`, `7`, and `4`.

## Fast reproduction using the selected parameters

The fitted `final_model.pkl` file is deliberately not versioned. Recreate it
from the data and saved selected parameters, then regenerate the figures:

```bash
python empirical/code/fit_selected_model.py
python empirical/code/make_final_figures.py
```

## Complete cross-validation

```bash
python empirical/code/run_analysis.py
python empirical/code/make_final_figures.py
```

The union of all participants' predictor values is sorted and divided into five
contiguous blocks. For each fold, observations in the same x block are held out
for all participants. Within each fold and participant, the training-rating
mean and population SD (`ddof=0`, denominator equal to the training sample size)
are applied to both the training and held-out ratings. The validation criterion
is RMSE on that training-defined z scale, averaged equally across folds and
participants. After parameter selection, the final descriptive model is fitted
to all observations standardized within participant. The search evaluates 231 common hyperparameter
combinations: 21 logarithmically spaced length scales from `0.1` to `1.0` and
11 logarithmically spaced values of `beta` from `0.0002` to `0.02`.

## Historical standardization check (working record)

This check was removed from the manuscript and is retained for the analysis
record. It is not required to reproduce the primary results.

`run_standardization_sensitivity.py` compares the primary procedure with the
historical approach of standardizing all 50 ratings before cross-validation.
The latter selects `length = 0.2818382931` and `beta = 0.0050237729`. The
top-three participant-score subspaces from the two final fits have a mean
squared-cosine similarity of `0.998`; the four-cluster partition is identical.

```bash
python empirical/code/run_standardization_sensitivity.py
```

## Estimation and RKHS geometry

All empirical fits, including cross-validation and the standardization
sensitivity analysis, use the participant-specific dual KRR equation
`alpha_t = solve(K_tt + N_t * beta * I, s_t)`. The positive regularization term
provides the diagonal shift; no additional jitter is added. The function Gram
matrix is computed directly as `H[t, u] = alpha_t.T @ K_tu @ alpha_u`, using
the RBF cross-kernel matrix between the two participants' own observations.
PCA diagonalizes the centered function Gram matrix. Principal functions are
evaluated as normalized combinations of the centered fitted functions.
Ward clustering uses the distances derived from the complete function Gram
matrix. These calculations follow the manuscript's Method section and the
dual KRR and RKHS geometry used in the simulations.

The response transformations and the five contiguous validation blocks are
unchanged by the choice of linear solver. To reproduce a run, regenerate
`final_model.pkl` with the current code; the fitted model is a disposable
cache, and a pickle made with the previous common-basis implementation must
not be reused.

## Files

Run `python empirical/code/validate_model.py` from the release root to check
the implementation against an independent feature-space ridge solution and
PCA invariants, including unequal observation counts and multivariate inputs.
`python validate_release.py` also checks agreement between the complete
primary CV table and the training-fold branch of the sensitivity analysis.

- `data/`: 20 CSV files with 50 observations each; see `DATA_DICTIONARY.md`.
- `code/load_psycho_data2.py`: data loading and within-participant standardization.
- `code/run_analysis.py`: x-block CV, parameter selection, and final model fit.
- `code/run_standardization_sensitivity.py`: full-profile versus training-fold
  standardization sensitivity analysis.
- `code/fit_selected_model.py`: final model fit without repeating CV.
- `code/model/rkhs_pca.py`: participant-specific dual KRR and RKHS-PCA.
- `code/validate_model.py`: independent numerical checks of the model.
- `code/make_final_figures.py`: PCA, clustering, and manuscript figures.
- `results/xblock_cv_results.csv`: complete 231-condition CV table.
- `results/standardization_sensitivity_cv_results.csv`: complete two-procedure
  sensitivity table.
- `results/standardization_sensitivity_summary.json`: selected values and
  downstream stability summary for the two procedures.
- `results/selected_parameters.json`: selected parameters and CV definition.
- `results/contribution_ratios.csv`: RKHS-PCA variance proportions.
- `results/cluster_membership.csv`: descriptive four-group partition.
- `figures/`: vector figures generated by the final analysis.
