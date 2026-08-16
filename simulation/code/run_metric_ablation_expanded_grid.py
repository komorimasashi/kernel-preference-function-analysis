#!/usr/bin/env python3
"""Run the metric ablation with an expanded x-CV search grid."""

from pathlib import Path

import run_metric_ablation as ablation


# With N=50 on [-1, 1], these exponents correspond approximately to
# ell = 0.020, 0.032, 0.051, 0.081, 0.129, 0.205, 0.324, 0.514, 0.815.
ablation.ns["ELL_EXP_GRID"] = [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

# The original upper bound was lambda=1 and was selected in most replications.
ablation.ns["LAMBDA_EXP_GRID"] = [-4, -3, -2, -1, 0, 1, 2]

ablation.OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "simulation1_metric_ablation"
)


if __name__ == "__main__":
    ablation.main()
