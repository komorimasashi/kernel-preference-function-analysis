#!/usr/bin/env python3
"""Oracle latent-subspace recovery for KRR-L2-PCA and KRR-RKHS-PCA.

True functions are finite expansions in a known RBF RKHS.  After applying the
same subject-wise standardisation used in the manuscript, noiseless functions
define oracle RKHS and L2 principal subspaces.  The simulation evaluates how
well PCA applied to KRR estimates from noisy observations recovers those oracle
subspaces.  The kernel and length scale are fixed at their generating values;
only the common KRR regularisation is selected by x-direction CV.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import run_metric_ablation as common


np = common.np
pd = common.pd
plt = common.plt


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "results" / "simulation2_oracle_recovery"

N_SUBJECTS = 30
N_OBS = 50
N_REPEATS = 200
L_LIST = [1, 2, 3, 4, 5]
X_MIN, X_MAX = -1.0, 1.0
ELL_TRUE = 0.30
SNR = 1.0
LATENT_EIGENVALUES = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
LAMBDA_GRID = 10.0 ** np.arange(-4, 4, dtype=float)
XCV_K_FOLDS = 5
JITTER = 1e-6
RNG_SEED = 24680


def inverse_sqrt_spd(matrix, tolerance=1e-12):
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    threshold = max(float(values.max()), 1.0) * tolerance
    keep = values > threshold
    if not np.any(keep):
        raise ValueError("Matrix has no numerically positive eigenvalues")
    return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T


def make_latent_axes(x_basis, kernel_metric):
    """Construct five fixed, linearly independent functions in the RKHS."""
    n = len(x_basis)
    nearest = lambda location: int(np.argmin(np.abs(x_basis - location)))
    locations = [-0.80, -0.45, 0.0, 0.45, 0.80]
    e = []
    for location in locations:
        vector = np.zeros(n)
        vector[nearest(location)] = 1.0
        e.append(vector)

    # Localised and symmetric/asymmetric contrasts.  Each column is a finite
    # kernel expansion and hence belongs to the generating RKHS.
    seeds = np.column_stack(
        [
            e[0] - e[4],
            e[2] - 0.5 * (e[1] + e[3]),
            e[0] + e[4] - e[1] - e[3],
            e[0] - e[1] + e[3] - e[4],
            e[0] + e[1] + e[2] + e[3] + e[4],
        ]
    )
    gram = seeds.T @ kernel_metric @ seeds
    axes = seeds @ inverse_sqrt_spd(gram)
    check = axes.T @ kernel_metric @ axes
    if not np.allclose(check, np.eye(len(LATENT_EIGENVALUES)), atol=1e-7):
        raise RuntimeError("Failed to construct RKHS-orthonormal latent axes")
    return axes


def choose_lambda_by_xcv(x_obs, y_obs, ell, lambda_grid):
    folds = common.make_block_folds(len(x_obs), XCV_K_FOLDS)
    best_lambda = None
    best_score = -np.inf
    for value in lambda_grid:
        score = common.xcv_score_session_fast(
            x_obs,
            y_obs,
            "rbf",
            ell,
            float(value),
            folds,
            JITTER,
            "zRMSE",
        )
        if score > best_score:
            best_score = score
            best_lambda = float(value)
    return best_lambda, best_score


def metric_subspace_statistics(target_coefficients, estimate_coefficients, metric):
    """Principal-angle statistics between two coefficient subspaces."""
    # Coefficient matrices are (basis size x subspace dimension).
    cholesky = np.linalg.cholesky(metric)
    target_feature = cholesky.T @ target_coefficients
    estimate_feature = cholesky.T @ estimate_coefficients
    q_target, _ = np.linalg.qr(target_feature)
    q_estimate, _ = np.linalg.qr(estimate_feature)
    singular_values = np.linalg.svd(q_target.T @ q_estimate, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles = np.arccos(singular_values)
    return {
        "mean_cos2": float(np.mean(singular_values**2)),
        "rms_sin": float(np.sqrt(np.mean(1.0 - singular_values**2))),
        "max_angle_deg": float(np.degrees(np.max(angles))),
    }


def euclidean_subspace_statistics(target, estimate):
    q_target, _ = np.linalg.qr(target)
    q_estimate, _ = np.linalg.qr(estimate)
    singular_values = np.linalg.svd(q_target.T @ q_estimate, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return {
        "mean_cos2": float(np.mean(singular_values**2)),
        "rms_sin": float(np.sqrt(np.mean(1.0 - singular_values**2))),
    }


def component_alignment(target_coefficients, estimate_coefficients, metric):
    """Mean absolute cosine after the best component permutation."""
    target_gram = target_coefficients.T @ metric @ target_coefficients
    estimate_gram = estimate_coefficients.T @ metric @ estimate_coefficients
    target_normalised = target_coefficients @ inverse_sqrt_spd(target_gram)
    estimate_normalised = estimate_coefficients @ inverse_sqrt_spd(estimate_gram)
    cosine = np.abs(target_normalised.T @ metric @ estimate_normalised)
    dimension = cosine.shape[0]
    best = max(
        np.mean([cosine[index, permutation[index]] for index in range(dimension)])
        for permutation in permutations(range(dimension))
    )
    return float(best)


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


def simulate_replication(rng, x_obs, kernel_metric, l2_metric, latent_axes):
    latent_scores = rng.normal(size=(N_SUBJECTS, len(LATENT_EIGENVALUES)))
    latent_scores *= np.sqrt(LATENT_EIGENVALUES)[None, :]
    true_coefficients_raw = latent_scores @ latent_axes.T
    true_values_raw = true_coefficients_raw @ kernel_metric.T

    # The manuscript standardises each participant's responses.  Define the
    # oracle target by applying the same operation to noiseless functions.
    true_values_standardised = np.vstack([common.zscore(row) for row in true_values_raw])
    true_coefficients = common.solve_spd(kernel_metric, true_values_standardised.T).T

    observations = np.zeros_like(true_values_raw)
    for subject in range(N_SUBJECTS):
        noise_sd = np.std(true_values_raw[subject]) / SNR
        observations[subject] = true_values_raw[subject] + rng.normal(0.0, noise_sd, N_OBS)
    observations_standardised = np.vstack([common.zscore(row) for row in observations])

    selected_lambda, xcv_score = choose_lambda_by_xcv(
        x_obs, observations, ELL_TRUE, LAMBDA_GRID
    )
    estimated_coefficients = np.zeros((N_SUBJECTS, N_OBS))
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
        true_rkhs = pca_details(true_coefficients, kernel_metric, component_count)
        true_l2 = pca_details(true_coefficients, l2_metric, component_count)
        estimated_rkhs = pca_details(estimated_coefficients, kernel_metric, component_count)
        estimated_l2 = pca_details(estimated_coefficients, l2_metric, component_count)

        row = {
            "L": component_count,
            "lambda": selected_lambda,
            "xcv_score": xcv_score,
        }
        estimates = {"L2": estimated_l2, "RKHS": estimated_rkhs}
        targets = {
            "OracleRKHS": (true_rkhs, kernel_metric),
            "OracleL2": (true_l2, l2_metric),
        }
        for target_name, (target, target_metric) in targets.items():
            for estimate_name, estimate in estimates.items():
                function_stats = metric_subspace_statistics(
                    target["components"], estimate["components"], target_metric
                )
                score_stats = euclidean_subspace_statistics(
                    target["participant_vectors"], estimate["participant_vectors"]
                )
                prefix = f"{target_name}_{estimate_name}"
                row[f"{prefix}_function_mean_cos2"] = function_stats["mean_cos2"]
                row[f"{prefix}_function_rms_sin"] = function_stats["rms_sin"]
                row[f"{prefix}_function_max_angle_deg"] = function_stats["max_angle_deg"]
                row[f"{prefix}_score_mean_cos2"] = score_stats["mean_cos2"]
                row[f"{prefix}_score_rms_sin"] = score_stats["rms_sin"]
                row[f"{prefix}_component_cosine"] = component_alignment(
                    target["components"], estimate["components"], target_metric
                )
        rows.append(row)
    return rows


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def summarise(results):
    rows = []
    contrasts = []
    metrics = [
        "function_mean_cos2",
        "function_rms_sin",
        "function_max_angle_deg",
        "score_mean_cos2",
        "score_rms_sin",
        "component_cosine",
    ]
    for component_count, group in results.groupby("L"):
        for target in ("OracleRKHS", "OracleL2"):
            for method in ("L2", "RKHS"):
                for metric_name in metrics:
                    column = f"{target}_{method}_{metric_name}"
                    mean, low, high = mean_ci(group[column])
                    rows.append(
                        {
                            "L": component_count,
                            "target": target,
                            "method": method,
                            "metric": metric_name,
                            "mean": mean,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
            for metric_name in metrics:
                # Orient every contrast so that positive values favour RKHS.
                if metric_name in ("function_rms_sin", "function_max_angle_deg", "score_rms_sin"):
                    difference = (
                        group[f"{target}_L2_{metric_name}"]
                        - group[f"{target}_RKHS_{metric_name}"]
                    )
                else:
                    difference = (
                        group[f"{target}_RKHS_{metric_name}"]
                        - group[f"{target}_L2_{metric_name}"]
                    )
                mean, low, high = mean_ci(difference)
                contrasts.append(
                    {
                        "L": component_count,
                        "target": target,
                        "metric": metric_name,
                        "rkhs_advantage": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def plot_recovery(summary, target, metric_name, ylabel, basename):
    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    labels = {"L2": "KRR-L2-PCA", "RKHS": "KRR-RKHS-PCA"}
    markers = {"L2": "s", "RKHS": "^"}
    for method in ("L2", "RKHS"):
        values = summary[
            (summary["target"] == target)
            & (summary["method"] == method)
            & (summary["metric"] == metric_name)
        ]
        axis.errorbar(
            values["L"],
            values["mean"],
            yerr=np.vstack(
                [values["mean"] - values["ci_low"], values["ci_high"] - values["mean"]]
            ),
            marker=markers[method],
            capsize=3,
            label=labels[method],
        )
    axis.set_xlabel("Number of PCs (L)")
    axis.set_ylabel(ylabel)
    axis.set_xticks(L_LIST)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUT_DIR / f"{basename}.pdf")
    figure.savefig(OUT_DIR / f"{basename}.png", dpi=180)
    plt.close(figure)


def plot_advantage(contrasts):
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    specifications = [
        ("function_mean_cos2", "Function-subspace similarity"),
        ("score_mean_cos2", "Score-subspace similarity"),
    ]
    for axis, (metric_name, title) in zip(axes, specifications):
        values = contrasts[
            (contrasts["target"] == "OracleRKHS") & (contrasts["metric"] == metric_name)
        ]
        axis.axhline(0.0, color="black", linewidth=1)
        axis.errorbar(
            values["L"],
            values["rkhs_advantage"],
            yerr=np.vstack(
                [
                    values["rkhs_advantage"] - values["ci_low"],
                    values["ci_high"] - values["rkhs_advantage"],
                ]
            ),
            marker="o",
            capsize=3,
        )
        axis.set_xlabel("Number of PCs (L)")
        axis.set_ylabel("RKHS minus L2 similarity")
        axis.set_xticks(L_LIST)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT_DIR / "rkhs_advantage_oracle_subspace.pdf")
    figure.savefig(OUT_DIR / "rkhs_advantage_oracle_subspace.png", dpi=180)
    plt.close(figure)


def main():
    if pd is None:
        raise RuntimeError("pandas is required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_obs = np.linspace(X_MIN, X_MAX, N_OBS)
    # Use the exact RKHS metric; jitter is restricted to KRR linear solves.
    kernel_metric = common.k_rbf(x_obs, x_obs, ELL_TRUE)
    l2_metric = common.l2_metric_for_kernel_functions(
        x_obs, common.k_rbf, ELL_TRUE, n_quad=201
    ) + 1e-10 * np.eye(N_OBS)
    latent_axes = make_latent_axes(x_obs, kernel_metric)
    rng = np.random.default_rng(RNG_SEED)

    all_rows = []
    for replication in range(N_REPEATS):
        rows = simulate_replication(rng, x_obs, kernel_metric, l2_metric, latent_axes)
        all_rows.extend({"rep": replication, **row} for row in rows)
        if replication == 0 or (replication + 1) % 10 == 0:
            print(f"Completed {replication + 1}/{N_REPEATS}", flush=True)

    results = pd.DataFrame(all_rows)
    summary, contrasts = summarise(results)
    results.to_csv(OUT_DIR / "subspace_recovery_replication_results.csv", index=False)
    summary.to_csv(OUT_DIR / "subspace_recovery_summary.csv", index=False)
    contrasts.to_csv(OUT_DIR / "subspace_recovery_paired_contrasts.csv", index=False)

    plot_recovery(
        summary,
        "OracleRKHS",
        "function_mean_cos2",
        "Oracle RKHS function-subspace similarity",
        "oracle_rkhs_function_subspace_similarity",
    )
    plot_recovery(
        summary,
        "OracleRKHS",
        "score_mean_cos2",
        "Oracle RKHS score-subspace similarity",
        "oracle_rkhs_score_subspace_similarity",
    )
    plot_recovery(
        summary,
        "OracleL2",
        "function_mean_cos2",
        "Oracle L2 function-subspace similarity",
        "oracle_l2_function_subspace_similarity",
    )
    plot_advantage(contrasts)

    primary = summary[
        (summary["target"] == "OracleRKHS")
        & (summary["metric"].isin(["function_mean_cos2", "score_mean_cos2"]))
    ]
    print("\nMean recovery of the noiseless oracle RKHS subspace:")
    print(
        primary.pivot(index=["L", "metric"], columns="method", values="mean")
        .round(4)
        .to_string()
    )
    primary_contrasts = contrasts[
        (contrasts["target"] == "OracleRKHS")
        & (contrasts["metric"].isin(["function_mean_cos2", "score_mean_cos2"]))
    ]
    print("\nPaired RKHS advantage (positive favours KRR-RKHS-PCA):")
    print(primary_contrasts.round(4).to_string(index=False))
    print("\nSelected lambda frequencies:")
    print(results[results["L"] == 1]["lambda"].value_counts().sort_index().to_string())
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
