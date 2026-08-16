#!/usr/bin/env python3
"""Sensitivity analysis for recovery of a latent RKHS principal subspace.

The latent functions are fixed finite expansions over five kernel centres, so
the same population is used conceptually for every observation density.  Four
RKHS-orthonormal, integral-zero latent axes are generated.  For each N x SNR
condition, KRR-L2-PCA and KRR-RKHS-PCA use the same KRR fits.  Their recovery of
the noiseless oracle RKHS subspace is compared by principal angles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import run_metric_ablation as common
import run_latent_subspace_recovery as base


np = common.np
pd = common.pd
plt = common.plt


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "results" / "simulation2_sensitivity"

N_SUBJECTS = 30
N_REPEATS = 200
N_OBS_LIST = [15, 30, 50]
SNR_LIST = [0.5, 1.0, 2.0]
L_LIST = [1, 2, 3]
X_MIN, X_MAX = -1.0, 1.0
ELL_TRUE = 0.30
LATENT_EIGENVALUES = np.array([4.0, 2.0, 1.0, 0.5])
KERNEL_CENTRES = np.array([-0.80, -0.40, 0.0, 0.40, 0.80])
LAMBDA_GRID = 10.0 ** np.arange(-4, 4, dtype=float)
XCV_K_FOLDS = 5
JITTER = 1e-6
RNG_SEED = 13579
N_QUADRATURE = 401


def trapezoidal_grid(n=N_QUADRATURE):
    x = np.linspace(X_MIN, X_MAX, n)
    weights = np.full(n, (X_MAX - X_MIN) / (n - 1))
    weights[[0, -1]] *= 0.5
    return x, weights


def make_common_latent_axes():
    """Create four RKHS-orthonormal axes with zero continuous integral."""
    x_quad, weights = trapezoidal_grid()
    k_qc = common.k_rbf(x_quad, KERNEL_CENTRES, ELL_TRUE)
    integral_functionals = weights @ k_qc

    # The null space of one nonzero integral functional in R^5 has dimension 4.
    _, _, right_vectors = np.linalg.svd(integral_functionals[None, :], full_matrices=True)
    zero_integral_basis = right_vectors[1:, :].T

    k_cc = common.k_rbf(KERNEL_CENTRES, KERNEL_CENTRES, ELL_TRUE)
    gram = zero_integral_basis.T @ k_cc @ zero_integral_basis
    axes = zero_integral_basis @ base.inverse_sqrt_spd(gram)

    if not np.allclose(axes.T @ k_cc @ axes, np.eye(4), atol=1e-8):
        raise RuntimeError("Latent axes are not RKHS-orthonormal")
    if not np.allclose(integral_functionals @ axes, np.zeros(4), atol=1e-10):
        raise RuntimeError("Latent axes are not integral-zero")
    return axes


def l2_inner_product_blocks(x_obs):
    x_quad, weights = trapezoidal_grid()
    k_qc = common.k_rbf(x_quad, KERNEL_CENTRES, ELL_TRUE)
    k_qo = common.k_rbf(x_quad, x_obs, ELL_TRUE)
    m_cc = k_qc.T @ (weights[:, None] * k_qc)
    m_oo = k_qo.T @ (weights[:, None] * k_qo)
    m_co = k_qc.T @ (weights[:, None] * k_qo)
    return m_cc, m_oo, m_co


def cross_metric_subspace_statistics(
    target_coefficients,
    estimate_coefficients,
    target_metric,
    estimate_metric,
    cross_metric,
):
    """Principal angles for subspaces represented on different kernel bases."""
    target_gram = target_coefficients.T @ target_metric @ target_coefficients
    estimate_gram = estimate_coefficients.T @ estimate_metric @ estimate_coefficients
    target_normaliser = base.inverse_sqrt_spd(target_gram)
    estimate_normaliser = base.inverse_sqrt_spd(estimate_gram)
    cross = (
        target_normaliser
        @ target_coefficients.T
        @ cross_metric
        @ estimate_coefficients
        @ estimate_normaliser
    )
    singular_values = np.linalg.svd(cross, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return {
        "mean_cos2": float(np.mean(singular_values**2)),
        "rms_sin": float(np.sqrt(np.mean(1.0 - singular_values**2))),
        "max_angle_deg": float(np.degrees(np.max(np.arccos(singular_values)))),
    }


def pca_details(coefficients, metric, component_count):
    _, (scores, component_rows, eigenvalues, participant_vectors, _) = common.rkhs_pca(
        coefficients, metric, component_count
    )
    return {
        "scores": scores,
        "components": component_rows.T,
        "eigenvalues": eigenvalues,
        "participant_vectors": participant_vectors,
    }


def choose_lambda(x_obs, observations):
    folds = common.make_block_folds(len(x_obs), XCV_K_FOLDS)
    best_lambda = None
    best_score = -np.inf
    for candidate in LAMBDA_GRID:
        score = common.xcv_score_session_fast(
            x_obs,
            observations,
            "rbf",
            ELL_TRUE,
            float(candidate),
            folds,
            JITTER,
            "zRMSE",
        )
        if score > best_score:
            best_score = score
            best_lambda = float(candidate)
    return best_lambda, best_score


def simulate_replication(rng, n_obs, snr, latent_axes):
    x_obs = np.linspace(X_MIN, X_MAX, n_obs)
    k_cc = common.k_rbf(KERNEL_CENTRES, KERNEL_CENTRES, ELL_TRUE)
    k_oo = common.k_rbf(x_obs, x_obs, ELL_TRUE)
    k_co = common.k_rbf(KERNEL_CENTRES, x_obs, ELL_TRUE)
    k_oc = k_co.T
    m_cc, m_oo, m_co = l2_inner_product_blocks(x_obs)

    latent_scores = rng.normal(size=(N_SUBJECTS, len(LATENT_EIGENVALUES)))
    latent_scores *= np.sqrt(LATENT_EIGENVALUES)[None, :]
    true_coefficients_raw = latent_scores @ latent_axes.T

    # Continuous z-standardisation: the axes have zero integral, so only a
    # subject-specific scale change is needed and the functions remain in the
    # same four-dimensional RKHS span.
    x_quad, weights = trapezoidal_grid()
    k_qc = common.k_rbf(x_quad, KERNEL_CENTRES, ELL_TRUE)
    true_values_quad = true_coefficients_raw @ k_qc.T
    continuous_variance = (true_values_quad**2) @ weights / (X_MAX - X_MIN)
    scales = np.sqrt(np.maximum(continuous_variance, 1e-15))
    true_coefficients = true_coefficients_raw / scales[:, None]

    true_values_observed = true_coefficients_raw @ k_oc.T
    observations = np.zeros_like(true_values_observed)
    for subject in range(N_SUBJECTS):
        noise_sd = np.std(true_values_observed[subject]) / snr
        observations[subject] = true_values_observed[subject] + rng.normal(
            0.0, noise_sd, n_obs
        )
    observations_standardised = np.vstack([common.zscore(row) for row in observations])

    selected_lambda, xcv_score = choose_lambda(x_obs, observations)
    estimated_coefficients = np.zeros((N_SUBJECTS, n_obs))
    for subject in range(N_SUBJECTS):
        estimated_coefficients[subject] = common.krr_fit_coeffs_general(
            x_obs,
            observations_standardised[subject],
            x_obs,
            common.k_rbf,
            ELL_TRUE,
            selected_lambda,
            JITTER,
        )

    rows = []
    for component_count in L_LIST:
        oracle_rkhs = pca_details(true_coefficients, k_cc, component_count)
        oracle_l2 = pca_details(true_coefficients, m_cc, component_count)
        estimated_rkhs = pca_details(estimated_coefficients, k_oo, component_count)
        estimated_l2 = pca_details(estimated_coefficients, m_oo, component_count)

        row = {
            "L": component_count,
            "lambda": selected_lambda,
            "xcv_score": xcv_score,
        }
        for target_name, target, target_metric, target_cross in (
            ("OracleRKHS", oracle_rkhs, k_cc, k_co),
            ("OracleL2", oracle_l2, m_cc, m_co),
        ):
            for method_name, estimate, estimate_metric in (
                ("L2", estimated_l2, m_oo),
                ("RKHS", estimated_rkhs, k_oo),
            ):
                function_stats = cross_metric_subspace_statistics(
                    target["components"],
                    estimate["components"],
                    target_metric,
                    m_oo if target_name == "OracleL2" else k_oo,
                    target_cross,
                )
                score_stats = base.euclidean_subspace_statistics(
                    target["participant_vectors"], estimate["participant_vectors"]
                )
                prefix = f"{target_name}_{method_name}"
                row[f"{prefix}_function_mean_cos2"] = function_stats["mean_cos2"]
                row[f"{prefix}_function_rms_sin"] = function_stats["rms_sin"]
                row[f"{prefix}_function_max_angle_deg"] = function_stats["max_angle_deg"]
                row[f"{prefix}_score_mean_cos2"] = score_stats["mean_cos2"]
                row[f"{prefix}_score_rms_sin"] = score_stats["rms_sin"]
        rows.append(row)
    return rows


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def summarise(results):
    summary_rows = []
    contrast_rows = []
    metrics = [
        "function_mean_cos2",
        "function_rms_sin",
        "function_max_angle_deg",
        "score_mean_cos2",
        "score_rms_sin",
    ]
    grouping = ["N_obs", "SNR", "L"]
    for keys, group in results.groupby(grouping):
        n_obs, snr, component_count = keys
        for target in ("OracleRKHS", "OracleL2"):
            for method in ("L2", "RKHS"):
                for metric in metrics:
                    values = group[f"{target}_{method}_{metric}"]
                    mean, low, high = mean_ci(values)
                    summary_rows.append(
                        {
                            "N_obs": n_obs,
                            "SNR": snr,
                            "L": component_count,
                            "target": target,
                            "method": method,
                            "metric": metric,
                            "mean": mean,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
            for metric in metrics:
                if metric in ("function_rms_sin", "function_max_angle_deg", "score_rms_sin"):
                    advantage = group[f"{target}_L2_{metric}"] - group[f"{target}_RKHS_{metric}"]
                else:
                    advantage = group[f"{target}_RKHS_{metric}"] - group[f"{target}_L2_{metric}"]
                mean, low, high = mean_ci(advantage)
                contrast_rows.append(
                    {
                        "N_obs": n_obs,
                        "SNR": snr,
                        "L": component_count,
                        "target": target,
                        "metric": metric,
                        "rkhs_advantage": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def plot_heatmap(contrasts, metric, title, basename):
    values = contrasts[
        (contrasts["target"] == "OracleRKHS")
        & (contrasts["metric"] == metric)
        & (contrasts["L"] == 3)
    ].copy()
    matrix = values.pivot(index="SNR", columns="N_obs", values="rkhs_advantage").loc[
        SNR_LIST, N_OBS_LIST
    ]
    low = values.pivot(index="SNR", columns="N_obs", values="ci_low").loc[
        SNR_LIST, N_OBS_LIST
    ]
    high = values.pivot(index="SNR", columns="N_obs", values="ci_high").loc[
        SNR_LIST, N_OBS_LIST
    ]

    limit = max(abs(float(matrix.values.min())), abs(float(matrix.values.max())), 1e-6)
    figure, axis = plt.subplots(figsize=(6.3, 4.4))
    image = axis.imshow(matrix.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row_index, snr in enumerate(SNR_LIST):
        for column_index, n_obs in enumerate(N_OBS_LIST):
            significant = low.loc[snr, n_obs] > 0 or high.loc[snr, n_obs] < 0
            suffix = "*" if significant else ""
            axis.text(
                column_index,
                row_index,
                f"{matrix.loc[snr, n_obs]:.3f}{suffix}",
                ha="center",
                va="center",
                color="black",
            )
    axis.set_xticks(range(len(N_OBS_LIST)), labels=N_OBS_LIST)
    axis.set_yticks(range(len(SNR_LIST)), labels=SNR_LIST)
    axis.set_xlabel("Observations per participant (N)")
    axis.set_ylabel("SNR")
    axis.set_title(title + " (L=3)")
    colourbar = figure.colorbar(image, ax=axis)
    colourbar.set_label("RKHS minus L2 similarity")
    axis.text(
        0.0,
        -0.18,
        "* paired 95% CI excludes zero; positive values favour RKHS",
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(OUT_DIR / f"{basename}.pdf")
    figure.savefig(OUT_DIR / f"{basename}.png", dpi=180)
    plt.close(figure)


def main():
    if pd is None:
        raise RuntimeError("pandas is required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latent_axes = make_common_latent_axes()
    all_rows = []
    condition_index = 0
    for n_obs in N_OBS_LIST:
        for snr in SNR_LIST:
            condition_seed = RNG_SEED + 100_000 * condition_index
            rng = np.random.default_rng(condition_seed)
            print(f"Starting N={n_obs}, SNR={snr}", flush=True)
            for replication in range(N_REPEATS):
                rows = simulate_replication(rng, n_obs, snr, latent_axes)
                all_rows.extend(
                    {
                        "N_obs": n_obs,
                        "SNR": snr,
                        "rep": replication,
                        **row,
                    }
                    for row in rows
                )
                if replication == 0 or (replication + 1) % 50 == 0:
                    print(
                        f"  N={n_obs}, SNR={snr}: {replication + 1}/{N_REPEATS}",
                        flush=True,
                    )
            condition_index += 1

    results = pd.DataFrame(all_rows)
    summary, contrasts = summarise(results)
    results.to_csv(OUT_DIR / "sensitivity_replication_results.csv", index=False)
    summary.to_csv(OUT_DIR / "sensitivity_summary.csv", index=False)
    contrasts.to_csv(OUT_DIR / "sensitivity_paired_contrasts.csv", index=False)

    plot_heatmap(
        contrasts,
        "function_mean_cos2",
        "Oracle RKHS function-subspace recovery",
        "sensitivity_function_subspace_L3",
    )
    plot_heatmap(
        contrasts,
        "score_mean_cos2",
        "Oracle RKHS score-subspace recovery",
        "sensitivity_score_subspace_L3",
    )

    primary = contrasts[
        (contrasts["target"] == "OracleRKHS")
        & (contrasts["metric"].isin(["function_mean_cos2", "score_mean_cos2"]))
        & (contrasts["L"] == 3)
    ]
    print("\nPrimary L=3 paired RKHS advantages:")
    print(primary.round(4).to_string(index=False))
    print("\nSelected lambda frequencies by condition:")
    print(
        results[results["L"] == 1]
        .groupby(["N_obs", "SNR", "lambda"])
        .size()
        .to_string()
    )
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
