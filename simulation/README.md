# Simulation Materials

## Code

- `simulation_common.py`: shared RBF-KRR, PCA, and x-block CV routines.
- `run_metric_ablation.py`: final expanded-grid Simulation 1 analysis.
- `run_latent_subspace_sensitivity.py`: final `T=30` Simulation 2 analysis.
- `run_latent_subspace_sensitivity_T20_N50.py`: additional analysis matching
  the empirical sample design.
- `make_manuscript_simulation_figures.py`: regenerates the two simulation
  figures from the saved result tables.

## Saved results

- `results/simulation1_metric_ablation/`: final expanded-grid Simulation 1
  results used in the manuscript.
- `results/simulation2_sensitivity/`: final `T=30` Simulation 2 results used in
  the function-subspace figure. The `oracle_l2_reference_*` tables give the
  noiseless similarity between the RKHS-PCA target and the L2-PCA subspace for
  the same Monte Carlo replications. Because this reference does not depend on
  `N` or SNR, its summary pools all 1,800 replications within each retained
  dimension `L`. It describes the difference between the two metrics when
  the noiseless functions are supplied directly to PCA; it is not a
  finite-sample upper bound on estimation accuracy.
- `results/simulation2_T20_N50/`: 200-replication sensitivity analysis matching
  the empirical `T=20, N=50` design. It is briefly reported in the main
  manuscript as a sample-size sensitivity analysis.

In both simulations, each participant's training-fold mean and population SD
are used to standardize both training and held-out responses. Held-out
responses and KRR predictions are compared directly on that same z scale,
without independently restandardizing predictions. The selection criterion
averages RMSE equally across participants and folds, retaining the shrinkage
effect of the regularization parameter. After selecting common parameters
within each replication, the final fit uses all observations standardized
within participant. KRR uses the exact dual system
`(K_train + n_train * beta * I) alpha = standardized_training_responses`.
Simulation 1 searches nine
length scales of the form `h * 10**a`, where `h = 2 / (N - 1)` and
`a = -0.3, -0.1, ..., 1.3`, together with regularization values from `1e-4` to
`1e2`. Simulation 2 fixes the oracle length scale and searches regularization
values from `1e-4` to `1e3`. The Simulation 2 result directories also contain
`regularization_selection_summary.csv`. The upper boundary is selected in
0.5% of replications at `N=15, SNR=0.5` and in none of the other conditions.

Simulation 2 targets the principal subspace of the finite set of noiseless,
continuously standardized functions in each replication, rather than the
population eigenfunctions or the original unstandardized latent axes. For
the RKHS target, both estimated subspaces are evaluated with the RKHS inner
product; for the L2 target, both are evaluated with the L2 inner product.
Among the 27 `N, SNR, L` combinations for the RKHS target, the RKHS method
has higher mean similarity in 25. At `N=15, SNR=1, L=3`, the means are close
(L2: 0.743; RKHS: 0.739). At `N=50, SNR=2, L=3`, the L2 method is more
accurate (L2: 0.819; RKHS: 0.759); this reversal also occurs with `T=20`
(L2: 0.821; RKHS: 0.735). Thus greater observation density and SNR do not
guarantee monotonic improvement or an advantage for the matching PCA metric.

The original exploratory notebook, pilot runs, earlier narrow-grid analysis,
score-subspace analysis, and intermediate diagnostics are not included.
