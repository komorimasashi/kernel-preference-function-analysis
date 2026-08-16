# Simulation Materials

## Code

- `run_metric_ablation_expanded_grid.py`: final Simulation 1 analysis.
- `run_metric_ablation.py`: shared implementation for Simulation 1 and common
  KRR/PCA utilities used by Simulation 2.
- `sim_rkhs_shape_vs_pca_dense_xcv_Goversion.ipynb`: definitions inherited by
  the Simulation 1 implementation.
- `run_latent_subspace_sensitivity.py`: final `T=30` Simulation 2 analysis.
- `run_latent_subspace_sensitivity_T20_N50.py`: additional analysis matching
  the empirical sample design.
- `run_latent_subspace_recovery.py`: shared latent-subspace utilities.
- `make_manuscript_simulation_figures.py`: regenerates the four simulation
  figures from the saved result tables.

## Saved results

- `results/simulation1_metric_ablation/`: final expanded-grid Simulation 1
  results used in the manuscript.
- `results/simulation2_sensitivity/`: final `T=30` Simulation 2 results used in
  the main figure and appendix figure.
- `results/simulation2_T20_N50/`: 200-replication sensitivity analysis matching
  the empirical `T=20, N=50` design.

Pilot runs, the earlier narrow-grid analysis, and intermediate diagnostics are
not included.
