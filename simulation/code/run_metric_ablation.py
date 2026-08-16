#!/usr/bin/env python3
"""Metric ablation for the KRR-PCA simulation.

The original notebook compares Raw-PCA with KRR -> RKHS-PCA.  This script adds
KRR -> L2-PCA while keeping the generated data, KRR fits, selected
hyperparameters, number of PCs, and evaluation grid common across the two KRR
pipelines.  The L2 inner product is approximated by trapezoidal quadrature on a
201-point grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "sim_rkhs_shape_vs_pca_dense_xcv_Goversion.ipynb"
OUT_DIR = HERE.parent / "results" / "simulation1_metric_ablation_base"


def load_original_definitions():
    """Execute configuration/import/function cells 1--7 from the notebook."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "metric_ablation_notebook_defs"}
    for cell_index in range(1, 8):
        source = "".join(notebook["cells"][cell_index].get("source", []))
        exec(compile(source, f"{NOTEBOOK}:cell-{cell_index}", "exec"), namespace)
    return namespace


ns = load_original_definitions()
np = ns["np"]
npl = ns["npl"]
pd = ns["pd"]
plt = ns["plt"]

k_rbf = ns["k_rbf"]
KERNELS = ns["KERNELS"]
solve_spd = ns["solve_spd"]
zscore = ns["zscore"]
rmse = ns["rmse"]
crmse = ns["crmse"]
crmse_z = ns["crmse_z"]
safe_corr = ns["safe_corr"]
make_block_folds = ns["make_block_folds"]
krr_fit_coeffs_general = ns["krr_fit_coeffs_general"]
metrics_shape_grid = ns["metrics_shape_grid"]
reconstruct_from_dual_pca = ns["reconstruct_from_dual_pca"]
rkhs_pca = ns["rkhs_pca"]


def _row_zscore(x):
    means = x.mean(axis=1, keepdims=True)
    sds = x.std(axis=1, keepdims=True)
    sds = np.where(sds == 0.0, 1.0, sds)
    return (x - means) / sds


def xcv_score_session_fast(x_obs, y_obs, kernel_name, ell, lam, folds, jitter, objective):
    """Same session-level x-CV score as the notebook, with subjects as RHSs."""
    kernel_fn = KERNELS[kernel_name]
    n = len(x_obs)
    k_bb = kernel_fn(x_obs, x_obs, ell) + jitter * np.eye(n)
    fold_scores = []

    for test_index in folds:
        train_index = np.setdiff1d(np.arange(n), test_index)
        y_train = y_obs[:, train_index]
        means = y_train.mean(axis=1, keepdims=True)
        sds = y_train.std(axis=1, keepdims=True)
        sds = np.where(sds == 0.0, 1.0, sds)
        s_train = (y_train - means) / sds

        k_xb = kernel_fn(x_obs[train_index], x_obs, ell)
        system = k_xb.T @ k_xb + (len(train_index) * lam) * k_bb
        coefficients = solve_spd(system, k_xb.T @ s_train.T).T
        s_prediction = (kernel_fn(x_obs[test_index], x_obs, ell) @ coefficients.T).T
        y_prediction = means + sds * s_prediction
        y_test = y_obs[:, test_index]

        if objective == "shape_R2":
            a = _row_zscore(y_test)
            b = _row_zscore(y_prediction)
            correlations = np.mean(a * b, axis=1)
            fold_scores.extend((correlations**2).tolist())
        elif objective in ("zRMSE", "cRMSE_z"):
            errors = np.sqrt(np.mean((_row_zscore(y_test) - _row_zscore(y_prediction)) ** 2, axis=1))
            fold_scores.extend((-errors).tolist())
        elif objective == "RMSE":
            errors = np.sqrt(np.mean((y_test - y_prediction) ** 2, axis=1))
            fold_scores.extend((-errors).tolist())
        elif objective == "cRMSE":
            differences = y_test - y_prediction
            centered = differences - differences.mean(axis=1, keepdims=True)
            errors = np.sqrt(np.mean(centered**2, axis=1))
            fold_scores.extend((-errors).tolist())
        else:
            raise ValueError(f"Unsupported x-CV objective: {objective}")

    return float(np.mean(fold_scores))


def xcv_select_session_params_fast(
    x_obs,
    y_obs,
    kernel_candidates,
    ell_exp_grid,
    lam_exp_grid,
    number_of_folds,
    jitter,
    objective,
):
    dx = np.diff(np.sort(x_obs)).mean() if len(x_obs) > 1 else 1.0
    ell_grid = (dx * (10.0 ** np.array(ell_exp_grid))).tolist()
    lam_grid = (10.0 ** np.array(lam_exp_grid)).tolist()
    folds = make_block_folds(len(x_obs), number_of_folds)
    best = {"score": -np.inf}
    for kernel_name in kernel_candidates:
        for ell in ell_grid:
            for lam in lam_grid:
                score = xcv_score_session_fast(
                    x_obs, y_obs, kernel_name, ell, lam, folds, jitter, objective
                )
                if score > best["score"]:
                    best = {
                        "score": score,
                        "kernel": kernel_name,
                        "ell": float(ell),
                        "lam": float(lam),
                    }
    return best


def l2_metric_for_kernel_functions(x_basis, kernel_fn, ell, n_quad=201):
    """Approximate integral k(x,X)^T k(x,X) dx by trapezoidal quadrature."""
    x_quad = np.linspace(float(x_basis.min()), float(x_basis.max()), n_quad)
    weights = np.full(n_quad, (x_quad[-1] - x_quad[0]) / (n_quad - 1))
    weights[[0, -1]] *= 0.5
    k_qb = kernel_fn(x_quad, x_basis, ell)
    metric = k_qb.T @ (weights[:, None] * k_qb)
    return 0.5 * (metric + metric.T)


def simulate_one_replication(x_obs, x_eval, config):
    t_count = config["n_subjects"]
    n_obs = len(x_obs)
    n_eval = len(x_eval)

    f_true_obs = np.zeros((t_count, n_obs))
    f_true_eval = np.zeros((t_count, n_eval))
    if config["generation_mode"] == "rkhs":
        k_gen_obs = k_rbf(x_obs, x_obs, config["ell_true"])
        k_gen_eval = k_rbf(x_eval, x_obs, config["ell_true"])
        for subject in range(t_count):
            coefficients = np.random.normal(0.0, config["sigma_a"], size=n_obs)
            f_true_obs[subject] = k_gen_obs @ coefficients
            f_true_eval[subject] = k_gen_eval @ coefficients
    elif config["generation_mode"] == "gp":
        x_joint = np.concatenate([x_obs, x_eval])
        k_joint = k_rbf(x_joint, x_joint, config["ell_true"]) + 1e-10 * np.eye(len(x_joint))
        cholesky = np.linalg.cholesky(k_joint)
        for subject in range(t_count):
            f_joint = cholesky @ np.random.normal(size=len(x_joint))
            f_true_obs[subject] = f_joint[:n_obs]
            f_true_eval[subject] = f_joint[n_obs:]
    else:
        raise ValueError(f"Unknown generation mode: {config['generation_mode']}")

    y_obs = np.zeros((t_count, n_obs))
    for subject in range(t_count):
        noise_std = np.std(f_true_obs[subject]) / max(config["snr"], 1e-12)
        y_obs[subject] = f_true_obs[subject] + np.random.normal(0.0, noise_std, size=n_obs)

    best = xcv_select_session_params_fast(
        x_obs,
        y_obs,
        config["kernel_candidates"],
        config["ell_exp_grid"],
        config["lam_exp_grid"],
        config["xcv_k"],
        config["jitter"],
        config["xcv_objective"],
    )
    kernel_fn = KERNELS[best["kernel"]]
    # The RKHS inner product uses the kernel itself.  Jitter belongs only in
    # the numerical linear solves used to fit KRR, not in the PCA metric.
    k_basis = kernel_fn(x_obs, x_obs, best["ell"])
    k_eval_basis = kernel_fn(x_eval, x_obs, best["ell"])
    l2_metric = l2_metric_for_kernel_functions(x_obs, kernel_fn, best["ell"])

    y_standardized = np.vstack([zscore(row) for row in y_obs])
    a_hat = np.zeros((t_count, n_obs))
    for subject in range(t_count):
        a_hat[subject] = krr_fit_coeffs_general(
            x_obs,
            y_standardized[subject],
            x_obs,
            kernel_fn,
            best["ell"],
            best["lam"],
            config["jitter"],
        )

    rows = []
    for component_count in config["l_list"]:
        raw_reconstruction, _ = reconstruct_from_dual_pca(y_standardized, component_count)
        raw_eval = np.vstack(
            [np.interp(x_eval, x_obs, raw_reconstruction[subject]) for subject in range(t_count)]
        )

        l2_coefficients, _ = rkhs_pca(a_hat, l2_metric, component_count)
        l2_eval = l2_coefficients @ k_eval_basis.T

        rkhs_coefficients, _ = rkhs_pca(a_hat, k_basis, component_count)
        rkhs_eval = rkhs_coefficients @ k_eval_basis.T

        estimates = {"Raw": raw_eval, "L2": l2_eval, "RKHS": rkhs_eval}
        row = {
            "L": component_count,
            "kernel": best["kernel"],
            "ell": best["ell"],
            "lam_session": best["lam"],
            "xcv_score_selected": best["score"],
        }
        for condition, estimate in estimates.items():
            subject_metrics = [
                metrics_shape_grid(x_eval, f_true_eval[subject], estimate[subject])
                for subject in range(t_count)
            ]
            for metric in ("shape_R2", "RMSE", "cRMSE_z", "zRMSE", "dX_peak"):
                values = [entry[metric] for entry in subject_metrics]
                row[f"{condition}_{metric}_mean"] = float(np.mean(values))
                row[f"{condition}_{metric}_sd"] = float(np.std(values, ddof=1))
        rows.append(row)
    return rows


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def make_summaries(results):
    summary_rows = []
    difference_rows = []
    conditions = ("Raw", "L2", "RKHS")
    metrics = ("shape_R2", "RMSE", "cRMSE_z", "zRMSE", "dX_peak")
    for component_count, group in results.groupby("L"):
        for condition in conditions:
            for metric in metrics:
                mean, low, high = mean_ci(group[f"{condition}_{metric}_mean"])
                summary_rows.append(
                    {
                        "L": component_count,
                        "condition": condition,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
        for metric in ("zRMSE", "dX_peak", "RMSE", "cRMSE_z"):
            difference = group[f"L2_{metric}_mean"] - group[f"RKHS_{metric}_mean"]
            mean, low, high = mean_ci(difference)
            difference_rows.append(
                {
                    "L": component_count,
                    "metric": metric,
                    "contrast": "KRR-L2 minus KRR-RKHS",
                    "mean_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "positive_favors": "KRR-RKHS",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(difference_rows)


def plot_metric(summary, metric, ylabel, basename):
    labels = {
        "Raw": "Raw-PCA",
        "L2": "KRR-L2-PCA",
        "RKHS": "KRR-RKHS-PCA",
    }
    markers = {"Raw": "o", "L2": "s", "RKHS": "^"}
    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    for condition in ("Raw", "L2", "RKHS"):
        values = summary[(summary["condition"] == condition) & (summary["metric"] == metric)]
        axis.errorbar(
            values["L"],
            values["mean"],
            yerr=np.vstack(
                [values["mean"] - values["ci_low"], values["ci_high"] - values["mean"]]
            ),
            marker=markers[condition],
            capsize=3,
            label=labels[condition],
        )
    axis.set_xlabel("Number of PCs (L)")
    axis.set_ylabel(ylabel)
    axis.set_xticks(sorted(summary["L"].unique()))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUT_DIR / f"{basename}.pdf")
    figure.savefig(OUT_DIR / f"{basename}.png", dpi=180)
    plt.close(figure)


def plot_paired_differences(differences):
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    specifications = (("zRMSE", "z-RMSE"), ("dX_peak", "Peak-location error"))
    for axis, (metric, label) in zip(axes, specifications):
        values = differences[differences["metric"] == metric]
        axis.axhline(0.0, color="black", linewidth=1)
        axis.errorbar(
            values["L"],
            values["mean_difference"],
            yerr=np.vstack(
                [
                    values["mean_difference"] - values["ci_low"],
                    values["ci_high"] - values["mean_difference"],
                ]
            ),
            marker="o",
            capsize=3,
        )
        axis.set_xlabel("Number of PCs (L)")
        axis.set_ylabel(f"L2 minus RKHS: {label}")
        axis.set_xticks(sorted(values["L"].unique()))
        axis.grid(alpha=0.25)
        axis.set_title("Positive values favor RKHS")
    figure.tight_layout()
    figure.savefig(OUT_DIR / "paired_metric_effect.pdf")
    figure.savefig(OUT_DIR / "paired_metric_effect.png", dpi=180)
    plt.close(figure)


def main():
    if pd is None:
        raise RuntimeError("pandas is required")
    if ns["LAMBDA_SELECTION_MODE"] != "session_shared":
        raise RuntimeError("This ablation is set up for the notebook's session_shared mode")

    config = {
        "n_subjects": ns["N_SUBJECTS"],
        "n_repeats": ns["N_REPEATS"],
        "l_list": ns["L_LIST"],
        "generation_mode": ns["GENERATION_MODE"],
        "ell_true": ns["ELL_TRUE"],
        "sigma_a": ns["SIGMA_A"],
        "snr": ns["SNR"],
        "kernel_candidates": ns["KERNEL_CANDIDATES"],
        "ell_exp_grid": ns["ELL_EXP_GRID"],
        "lam_exp_grid": ns["LAMBDA_EXP_GRID"],
        "xcv_k": ns["XCV_K_FOLDS"],
        "xcv_objective": ns["XCV_OBJECTIVE"],
        "jitter": ns["JITTER"],
    }
    x_obs = np.linspace(ns["X_MIN"], ns["X_MAX"], ns["N_OBS_PER_SUBJECT"])
    x_eval = np.linspace(ns["X_MIN"], ns["X_MAX"], ns["M_EVAL"])
    np.random.seed(ns["RNG_SEED"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for replication in range(config["n_repeats"]):
        rows = simulate_one_replication(x_obs, x_eval, config)
        all_rows.extend({"rep": replication, **row} for row in rows)
        if (replication + 1) % 10 == 0 or replication == 0:
            print(f"Completed {replication + 1}/{config['n_repeats']}", flush=True)

    results = pd.DataFrame(all_rows)
    summary, differences = make_summaries(results)
    results.to_csv(OUT_DIR / "metric_ablation_replication_results.csv", index=False)
    summary.to_csv(OUT_DIR / "metric_ablation_summary.csv", index=False)
    differences.to_csv(OUT_DIR / "metric_ablation_paired_differences.csv", index=False)

    plot_metric(summary, "zRMSE", "z-RMSE (lower is better)", "metric_ablation_zRMSE_vs_L")
    plot_metric(
        summary,
        "dX_peak",
        "Peak-location error (lower is better)",
        "metric_ablation_peak_vs_L",
    )
    plot_metric(summary, "shape_R2", "Shape R-squared (higher is better)", "metric_ablation_shape_R2_vs_L")
    plot_paired_differences(differences)

    print("\nPrimary summaries:")
    print(
        summary[summary["metric"].isin(["zRMSE", "dX_peak"])]
        .pivot(index=["L", "metric"], columns="condition", values="mean")
        .round(4)
    )
    print("\nPaired KRR-L2 minus KRR-RKHS differences:")
    print(
        differences[differences["metric"].isin(["zRMSE", "dX_peak"])]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
