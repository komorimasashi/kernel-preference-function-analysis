#!/usr/bin/env python3
"""Empirical-design check: T=20, N=50, SNR in {0.5, 1, 2}."""

from pathlib import Path

import run_latent_subspace_sensitivity as sensitivity


sensitivity.N_SUBJECTS = 20
sensitivity.N_OBS_LIST = [50]
sensitivity.SNR_LIST = [0.5, 1.0, 2.0]
sensitivity.N_REPEATS = 200
sensitivity.OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "simulation2_T20_N50"
)


if __name__ == "__main__":
    sensitivity.main()
