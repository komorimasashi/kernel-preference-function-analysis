#!/usr/bin/env python3
"""Simulation 2: recovery of latent RKHS and L2 function subspaces.

The latent functions are fixed expansions over five RBF-kernel centres. For
each observation-density and SNR condition, KRR-L2-PCA and KRR-RKHS-PCA share
the same KRR fits and differ only in the PCA metric. The reported outcome is
mean squared cosine similarity between true and estimated function subspaces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import simulation_common as common


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "simulation2_sensitivity"

N_SUBJECTS = 30
N_REPEATS = 200
N_OBS_LIST = [15, 30, 50]
SNR_LIST = [0.5, 1.0, 2.0]
L_LIST = [1, 2, 3]
X_MIN, X_MAX = -1.0, 1.0
ELL_TRUE = 0.30
LATENT_EIGENVALUES = np.array([4.0, 2.0, 1.0, 0.5])
KERNEL_CENTRES = np.array([-0.80, -0.40, 0.0, 0.40, 0.80])
REGULARIZATION_GRID = 10.0 ** np.arange(-4, 4, dtype=float)
XCV_FOLDS = 5
RNG_SEED = 13579
N_QUADRATURE = 401
SAVE_ORACLE_REFERENCE = True


def inverse_sqrt_spd(matrix, tolerance=1e-12):
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    threshold = max(float(values.max()), 1.0) * tolerance
    keep = values > threshold
    if not np.any(keep):
        raise ValueError("Matrix has no numerically positive eigenvalues")
    return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T


def trapezoidal_grid(point_count=N_QUADRATURE):
    x = np.linspace(X_MIN, X_MAX, point_count)
    weights = np.full(point_count, (X_MAX - X_MIN) / (point_count - 1))
    weights[[0, -1]] *= 0.5
    return x, weights


def make_common_latent_axes():
    """Create four RKHS-orthonormal axes with zero continuous integral."""
    x_quad, weights = trapezoidal_grid()
    k_quad_centres = common.k_rbf(x_quad, KERNEL_CENTRES, ELL_TRUE)
    integral_functionals = weights @ k_quad_centres

    _, _, right_vectors = np.linalg.svd(
        integral_functionals[None, :], full_matrices=True
    )
    zero_integral_basis = right_vectors[1:, :].T
    k_centres = common.k_rbf(KERNEL_CENTRES, KERNEL_CENTRES, ELL_TRUE)
    gram = zero_integral_basis.T @ k_centres @ zero_integral_basis
    axes = zero_integral_basis @ inverse_sqrt_spd(gram)

    if not np.allclose(axes.T @ k_centres @ axes, np.eye(4), atol=1e-8):
        raise RuntimeError("Latent axes are not RKHS-orthonormal")
    if not np.allclose(integral_functionals @ axes, np.zeros(4), atol=1e-10):
        raise RuntimeError("Latent axes are not integral-zero")
    return axes


def l2_inner_product_blocks(x_obs):
    x_quad, weights = trapezoidal_grid()
    k_quad_centres = common.k_rbf(x_quad, KERNEL_CENTRES, ELL_TRUE)
    k_quad_observed = common.k_rbf(x_quad, x_obs, ELL_TRUE)
    centre_metric = k_quad_centres.T @ (weights[:, None] * k_quad_centres)
    observed_metric = k_quad_observed.T @ (weights[:, None] * k_quad_observed)
    cross_metric = k_quad_centres.T @ (weights[:, None] * k_quad_observed)
    return centre_metric, observed_metric, cross_metric


def cross_metric_mean_cos2(
    target_coefficients,
    estimate_coefficients,
    target_metric,
    estimate_metric,
    cross_metric,
):
    """Mean squared cosine between subspaces on different kernel bases."""
    target_gram = target_coefficients.T @ target_metric @ target_coefficients
    estimate_gram = estimate_coefficients.T @ estimate_metric @ estimate_coefficients
    cross = (
        inverse_sqrt_spd(target_gram)
        @ target_coefficients.T
        @ cross_metric
        @ estimate_coefficients
        @ inverse_sqrt_spd(estimate_gram)
    )
    singular_values = np.linalg.svd(cross, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return float(np.mean(singular_values**2))


def principal_components(coefficients, metric, component_count):
    _, component_rows, _ = common.metric_pca(
        coefficients, metric, component_count
    )
    return component_rows.T


def choose_regularization(x_obs, observations):
    folds = common.make_block_folds(len(x_obs), XCV_FOLDS)
    best = {"score": -np.inf}
    for regularization in REGULARIZATION_GRID:
        score = common.xcv_zrmse_score_session(
            x_obs,
            observations,
            ELL_TRUE,
            float(regularization),
            folds,
        )
        if score > best["score"]:
            best = {
                "score": score,
                "regularization": float(regularization),
            }
    return best


def draw_true_coefficients(rng, latent_axes, k_quad_centres, weights):
    """Draw latent functions and their continuously standardized versions."""
    latent_scores = rng.normal(size=(N_SUBJECTS, len(LATENT_EIGENVALUES)))
    latent_scores *= np.sqrt(LATENT_EIGENVALUES)[None, :]
    raw_coefficients = latent_scores @ latent_axes.T

    true_values_quad = raw_coefficients @ k_quad_centres.T
    continuous_variance = (true_values_quad**2) @ weights / (X_MAX - X_MIN)
    scales = np.sqrt(np.maximum(continuous_variance, 1e-15))
    standardized_coefficients = raw_coefficients / scales[:, None]
    return raw_coefficients, standardized_coefficients


def generate_noisy_observations(rng, true_values_observed, snr):
    """Add subject-specific Gaussian noise at the requested SNR."""
    observations = np.zeros_like(true_values_observed)
    for subject in range(N_SUBJECTS):
        noise_sd = np.std(true_values_observed[subject]) / snr
        observations[subject] = true_values_observed[subject] + rng.normal(
            0.0, noise_sd, true_values_observed.shape[1]
        )
    return observations


def simulate_replication(rng, observation_count, snr, latent_axes):
    x_obs = np.linspace(X_MIN, X_MAX, observation_count)
    k_centres = common.k_rbf(KERNEL_CENTRES, KERNEL_CENTRES, ELL_TRUE)
    k_observed = common.k_rbf(x_obs, x_obs, ELL_TRUE)
    k_centres_observed = common.k_rbf(KERNEL_CENTRES, x_obs, ELL_TRUE)
    centre_l2, observed_l2, cross_l2 = l2_inner_product_blocks(x_obs)

    x_quad, weights = trapezoidal_grid()
    k_quad_centres = common.k_rbf(x_quad, KERNEL_CENTRES, ELL_TRUE)
    true_coefficients_raw, true_coefficients = draw_true_coefficients(
        rng, latent_axes, k_quad_centres, weights
    )

    true_values_observed = true_coefficients_raw @ k_centres_observed
    observations = generate_noisy_observations(rng, true_values_observed, snr)
    standardized_observations = np.vstack(
        [common.zscore(row) for row in observations]
    )

    selected = choose_regularization(x_obs, observations)
    estimated_coefficients = np.vstack(
        [
            common.krr_fit_coeffs(
                x_obs,
                standardized_observations[subject],
                x_obs,
                ELL_TRUE,
                selected["regularization"],
            )
            for subject in range(N_SUBJECTS)
        ]
    )

    rows = []
    for component_count in L_LIST:
        targets = {
            "OracleRKHS": (
                principal_components(
                    true_coefficients, k_centres, component_count
                ),
                k_centres,
                k_centres_observed,
            ),
            "OracleL2": (
                principal_components(
                    true_coefficients, centre_l2, component_count
                ),
                centre_l2,
                cross_l2,
            ),
        }
        estimates = {
            "L2": principal_components(
                estimated_coefficients, observed_l2, component_count
            ),
            "RKHS": principal_components(
                estimated_coefficients, k_observed, component_count
            ),
        }

        row = {
            "L": component_count,
            "regularization": selected["regularization"],
            "xcv_score": selected["score"],
        }
        for target_name, (target, target_metric, target_cross) in targets.items():
            # Both estimated subspaces are evaluated in the target geometry.
            # They differ in how PCA selected them, not in the evaluation metric.
            estimate_metric = (
                k_observed if target_name == "OracleRKHS" else observed_l2
            )
            for method_name, estimate in estimates.items():
                row[f"{target_name}_{method_name}_function_mean_cos2"] = (
                    cross_metric_mean_cos2(
                        target,
                        estimate,
                        target_metric,
                        estimate_metric,
                        target_cross,
                    )
                )
        rows.append(row)
    return rows


def oracle_l2_reference_results():
    """Replay the design and compare noiseless L2 and RKHS PCA subspaces.

    The random-number stream is advanced through the observation-noise draws so
    that each reference value corresponds exactly to the Monte Carlo
    replication in the saved Simulation 2 results. No KRR fitting is required.
    """
    latent_axes = make_common_latent_axes()
    x_quad, weights = trapezoidal_grid()
    k_quad_centres = common.k_rbf(x_quad, KERNEL_CENTRES, ELL_TRUE)
    k_centres = common.k_rbf(KERNEL_CENTRES, KERNEL_CENTRES, ELL_TRUE)
    centre_l2 = k_quad_centres.T @ (weights[:, None] * k_quad_centres)

    rows = []
    condition_index = 0
    for observation_count in N_OBS_LIST:
        x_obs = np.linspace(X_MIN, X_MAX, observation_count)
        k_centres_observed = common.k_rbf(
            KERNEL_CENTRES, x_obs, ELL_TRUE
        )
        for snr in SNR_LIST:
            rng = np.random.default_rng(RNG_SEED + 100_000 * condition_index)
            for replication in range(N_REPEATS):
                raw_coefficients, true_coefficients = draw_true_coefficients(
                    rng, latent_axes, k_quad_centres, weights
                )
                true_values_observed = raw_coefficients @ k_centres_observed
                generate_noisy_observations(rng, true_values_observed, snr)

                for component_count in L_LIST:
                    rkhs_target = principal_components(
                        true_coefficients, k_centres, component_count
                    )
                    l2_target = principal_components(
                        true_coefficients, centre_l2, component_count
                    )
                    rows.append(
                        {
                            "N_obs": observation_count,
                            "SNR": snr,
                            "L": component_count,
                            "rep": replication,
                            "oracle_l2_to_rkhs_mean_cos2": (
                                cross_metric_mean_cos2(
                                    rkhs_target,
                                    l2_target,
                                    k_centres,
                                    k_centres,
                                    k_centres,
                                )
                            ),
                        }
                    )
            condition_index += 1
    return pd.DataFrame(rows)


def summarise_oracle_l2_reference(reference_results):
    rows = []
    # The noiseless reference distribution is defined by the latent function
    # generator and L; it does not depend on the observation count or SNR.
    # Pool all nine design cells to avoid displaying Monte Carlo variation as
    # an apparent dependence of the reference on N or SNR.
    for component_count, group in reference_results.groupby("L"):
        mean, low, high = mean_ci(group["oracle_l2_to_rkhs_mean_cos2"])
        rows.append(
            {
                "L": component_count,
                "n_reps": len(group),
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def summarise(results):
    summary_rows = []
    contrast_rows = []
    for (observation_count, snr, component_count), group in results.groupby(
        ["N_obs", "SNR", "L"]
    ):
        for target in ("OracleRKHS", "OracleL2"):
            for method in ("L2", "RKHS"):
                values = group[f"{target}_{method}_function_mean_cos2"]
                mean, low, high = mean_ci(values)
                summary_rows.append(
                    {
                        "N_obs": observation_count,
                        "SNR": snr,
                        "L": component_count,
                        "target": target,
                        "method": method,
                        "metric": "function_mean_cos2",
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )

            advantage = (
                group[f"{target}_RKHS_function_mean_cos2"]
                - group[f"{target}_L2_function_mean_cos2"]
            )
            mean, low, high = mean_ci(advantage)
            contrast_rows.append(
                {
                    "N_obs": observation_count,
                    "SNR": snr,
                    "L": component_count,
                    "target": target,
                    "metric": "function_mean_cos2",
                    "rkhs_advantage": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def summarise_regularization_selection(results):
    """Summarise CV choices once per Monte Carlo replication."""
    selected = results.drop_duplicates(["N_obs", "SNR", "rep"])
    rows = []
    for (observation_count, snr), group in selected.groupby(["N_obs", "SNR"]):
        values = group["regularization"].to_numpy()
        rows.append(
            {
                "N_obs": observation_count,
                "SNR": snr,
                "n_reps": len(values),
                "median_regularization": float(np.median(values)),
                "proportion_at_lower_grid_boundary": float(
                    np.mean(values == REGULARIZATION_GRID[0])
                ),
                "proportion_at_upper_grid_boundary": float(
                    np.mean(values == REGULARIZATION_GRID[-1])
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latent_axes = make_common_latent_axes()
    rows = []
    condition_index = 0
    for observation_count in N_OBS_LIST:
        for snr in SNR_LIST:
            rng = np.random.default_rng(RNG_SEED + 100_000 * condition_index)
            print(f"Starting N={observation_count}, SNR={snr}", flush=True)
            for replication in range(N_REPEATS):
                rows.extend(
                    {
                        "N_obs": observation_count,
                        "SNR": snr,
                        "rep": replication,
                        **row,
                    }
                    for row in simulate_replication(
                        rng, observation_count, snr, latent_axes
                    )
                )
                if replication == 0 or (replication + 1) % 50 == 0:
                    print(
                        f"  N={observation_count}, SNR={snr}: "
                        f"{replication + 1}/{N_REPEATS}",
                        flush=True,
                    )
            condition_index += 1

    results = pd.DataFrame(rows)
    summary, contrasts = summarise(results)
    regularization_summary = summarise_regularization_selection(results)
    results.to_csv(OUT_DIR / "sensitivity_replication_results.csv", index=False)
    summary.to_csv(OUT_DIR / "sensitivity_summary.csv", index=False)
    contrasts.to_csv(OUT_DIR / "sensitivity_paired_contrasts.csv", index=False)
    regularization_summary.to_csv(
        OUT_DIR / "regularization_selection_summary.csv", index=False
    )
    if SAVE_ORACLE_REFERENCE:
        oracle_reference = oracle_l2_reference_results()
        oracle_reference_summary = summarise_oracle_l2_reference(oracle_reference)
        oracle_reference.to_csv(
            OUT_DIR / "oracle_l2_reference_replication_results.csv", index=False
        )
        oracle_reference_summary.to_csv(
            OUT_DIR / "oracle_l2_reference_summary.csv", index=False
        )

    primary = contrasts[
        (contrasts["target"] == "OracleRKHS") & (contrasts["L"] == 3)
    ]
    print(primary.round(4).to_string(index=False))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
