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
  the same Monte Carlo replications; this is an asymptotic reference rather
  than a deterministic finite-sample upper bound.
- `results/simulation2_T20_N50/`: 200-replication sensitivity analysis matching
  the empirical `T=20, N=50` design and supporting the corresponding sentence
  in the manuscript.

The original exploratory notebook, pilot runs, earlier narrow-grid analysis,
score-subspace analysis, and intermediate diagnostics are not included.
