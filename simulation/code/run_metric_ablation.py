#!/usr/bin/env python3
"""Simulation 1: metric ablation for pointwise function reconstruction.

Raw-PCA, KRR-L2-PCA, and KRR-RKHS-PCA are compared using the z-standardized
RMSE reported in the manuscript. The two KRR pipelines share generated data,
KRR fits, selected hyperparameters, and retained component counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import simulation_common as common


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "simulation1_metric_ablation"

N_SUBJECTS = 30
N_OBS_PER_SUBJECT = 50
N_REPEATS = 200
L_LIST = [1, 2, 3, 4, 5]
X_MIN, X_MAX = -1.0, 1.0
N_EVALUATION_POINTS = 21
ELL_TRUE = 0.30
SNR = 1.0

# Expanded final search grids. With N=50 on [-1, 1], the length-scale grid is
# approximately 0.020, 0.032, 0.051, 0.081, 0.129, 0.205, 0.324, 0.514, 0.815.
ELL_EXP_GRID = [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
REGULARIZATION_EXP_GRID = [-4, -3, -2, -1, 0, 1, 2]
XCV_FOLDS = 5
JITTER = 1e-6
RNG_SEED = 9999


def select_session_parameters(x_obs, y_obs):
    spacing = np.diff(np.sort(x_obs)).mean()
    ell_grid = spacing * (10.0 ** np.asarray(ELL_EXP_GRID))
    regularization_grid = 10.0 ** np.asarray(REGULARIZATION_EXP_GRID)
    folds = common.make_block_folds(len(x_obs), XCV_FOLDS)
    best = {"score": -np.inf}
    for ell in ell_grid:
        for regularization in regularization_grid:
            score = common.xcv_zrmse_score_session(
                x_obs,
                y_obs,
                float(ell),
                float(regularization),
                folds,
                JITTER,
            )
            if score > best["score"]:
                best = {
                    "score": score,
                    "ell": float(ell),
                    "regularization": float(regularization),
                }
    return best


def l2_metric_for_kernel_functions(x_basis, ell, quadrature_points=201):
    x_quad = np.linspace(float(x_basis.min()), float(x_basis.max()), quadrature_points)
    weights = np.full(
        quadrature_points, (x_quad[-1] - x_quad[0]) / (quadrature_points - 1)
    )
    weights[[0, -1]] *= 0.5
    k_quad_basis = common.k_rbf(x_quad, x_basis, ell)
    metric = k_quad_basis.T @ (weights[:, None] * k_quad_basis)
    return 0.5 * (metric + metric.T)


def zrmse(true_values, estimated_values):
    return float(
        np.sqrt(
            np.mean(
                (common.zscore(true_values) - common.zscore(estimated_values)) ** 2
            )
        )
    )


def simulate_one_replication(x_obs, x_eval):
    joint_x = np.concatenate([x_obs, x_eval])
    joint_kernel = common.k_rbf(joint_x, joint_x, ELL_TRUE)
    joint_kernel += 1e-10 * np.eye(len(joint_x))
    cholesky = np.linalg.cholesky(joint_kernel)

    true_observed = np.zeros((N_SUBJECTS, len(x_obs)))
    true_evaluation = np.zeros((N_SUBJECTS, len(x_eval)))
    for subject in range(N_SUBJECTS):
        function_values = cholesky @ np.random.normal(size=len(joint_x))
        true_observed[subject] = function_values[: len(x_obs)]
        true_evaluation[subject] = function_values[len(x_obs) :]

    observations = np.zeros_like(true_observed)
    for subject in range(N_SUBJECTS):
        noise_sd = np.std(true_observed[subject]) / SNR
        observations[subject] = true_observed[subject] + np.random.normal(
            0.0, noise_sd, len(x_obs)
        )

    selected = select_session_parameters(x_obs, observations)
    ell = selected["ell"]
    regularization = selected["regularization"]
    k_basis = common.k_rbf(x_obs, x_obs, ell)
    k_evaluation_basis = common.k_rbf(x_eval, x_obs, ell)
    l2_metric = l2_metric_for_kernel_functions(x_obs, ell)

    standardized_observations = np.vstack(
        [common.zscore(row) for row in observations]
    )
    fitted_coefficients = np.vstack(
        [
            common.krr_fit_coeffs(
                x_obs,
                standardized_observations[subject],
                x_obs,
                ell,
                regularization,
                JITTER,
            )
            for subject in range(N_SUBJECTS)
        ]
    )

    rows = []
    for component_count in L_LIST:
        raw_reconstruction = common.reconstruct_from_dual_pca(
            standardized_observations, component_count
        )
        raw_evaluation = np.vstack(
            [
                np.interp(x_eval, x_obs, raw_reconstruction[subject])
                for subject in range(N_SUBJECTS)
            ]
        )
        l2_coefficients, _, _ = common.metric_pca(
            fitted_coefficients, l2_metric, component_count
        )
        rkhs_coefficients, _, _ = common.metric_pca(
            fitted_coefficients, k_basis, component_count
        )
        estimates = {
            "Raw": raw_evaluation,
            "L2": l2_coefficients @ k_evaluation_basis.T,
            "RKHS": rkhs_coefficients @ k_evaluation_basis.T,
        }

        row = {
            "L": component_count,
            "ell": ell,
            "regularization": regularization,
            "xcv_score": selected["score"],
        }
        for condition, estimate in estimates.items():
            values = [
                zrmse(true_evaluation[subject], estimate[subject])
                for subject in range(N_SUBJECTS)
            ]
            row[f"{condition}_zRMSE_mean"] = float(np.mean(values))
            row[f"{condition}_zRMSE_sd"] = float(np.std(values, ddof=1))
        rows.append(row)
    return rows


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def summarise(results):
    rows = []
    for component_count, group in results.groupby("L"):
        for condition in ("Raw", "L2", "RKHS"):
            mean, low, high = mean_ci(group[f"{condition}_zRMSE_mean"])
            rows.append(
                {
                    "L": component_count,
                    "condition": condition,
                    "metric": "zRMSE",
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def main():
    x_obs = np.linspace(X_MIN, X_MAX, N_OBS_PER_SUBJECT)
    x_eval = np.linspace(X_MIN, X_MAX, N_EVALUATION_POINTS)
    np.random.seed(RNG_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for replication in range(N_REPEATS):
        rows.extend(
            {"rep": replication, **row}
            for row in simulate_one_replication(x_obs, x_eval)
        )
        if replication == 0 or (replication + 1) % 10 == 0:
            print(f"Completed {replication + 1}/{N_REPEATS}", flush=True)

    results = pd.DataFrame(rows)
    summary = summarise(results)
    results.to_csv(OUT_DIR / "metric_ablation_replication_results.csv", index=False)
    summary.to_csv(OUT_DIR / "metric_ablation_summary.csv", index=False)
    print(summary.pivot(index="L", columns="condition", values="mean").round(4))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
