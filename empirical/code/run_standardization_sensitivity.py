"""Compare response-standardization schemes for empirical x-block CV.

The primary comparison is between the historical procedure, which computes a
participant's mean and population SD from all 50 ratings before CV, and a
leakage-free procedure, which computes both quantities from the training fold
and applies that same transformation to the held-out fold.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from model.rkhs_pca import RKHSFunctionPCA, dual_krr_coefficients, rbf_kernel


PACKAGE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PACKAGE_DIR / "results"

STANDARDIZATION_SCHEMES = ("participant_full", "training_fold")
N_FOLDS = 5
LENGTH_GRID = np.logspace(-1, 0, 21)
BETA_GRID = np.logspace(np.log10(0.01 / 50.0), np.log10(1.0 / 50.0), 11)


def make_shared_x_block_masks(x_list: list[np.ndarray]) -> list[list[np.ndarray]]:
    x_union = np.unique(np.concatenate([x.reshape(-1) for x in x_list]))
    union_blocks = np.array_split(x_union, N_FOLDS)
    return [
        [np.isin(x.reshape(-1), block) for x in x_list]
        for block in union_blocks
    ]


def load_raw_data() -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for participant in range(1, 21):
        values = pd.read_csv(PACKAGE_DIR / "data" / f"data{participant}.csv")
        x_list.append(values["random_value"].to_numpy(dtype=np.float64).reshape(-1, 1))
        y_list.append(values["beauty"].to_numpy(dtype=np.float64))
    return x_list, y_list


def standardize(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    if sd == 0.0:
        raise ValueError("Cannot standardize ratings with zero variance.")
    return (values - mean) / sd


def evaluate_params(
    x_list: list[np.ndarray],
    raw_y_list: list[np.ndarray],
    validation_masks: list[list[np.ndarray]],
    length: float,
    beta: float,
    scheme: str,
) -> tuple[float, list[float]]:
    participant_fold_scores: list[list[float]] = [[] for _ in x_list]

    full_stats = None
    if scheme == "participant_full":
        full_stats = [(float(y.mean()), float(y.std(ddof=0))) for y in raw_y_list]
    elif scheme != "training_fold":
        raise ValueError(f"Unknown standardization scheme: {scheme}")

    for fold_masks in validation_masks:
        for participant, (x, y_raw, val_mask) in enumerate(
            zip(x_list, raw_y_list, fold_masks)
        ):
            train_mask = ~val_mask
            if not np.any(train_mask) or not np.any(val_mask):
                continue

            x_train = x[train_mask, 0]
            x_val = x[val_mask, 0]
            if full_stats is None:
                mean = float(y_raw[train_mask].mean())
                sd = float(y_raw[train_mask].std(ddof=0))
            else:
                mean, sd = full_stats[participant]
            y_train = standardize(y_raw[train_mask], mean, sd)
            y_val = standardize(y_raw[val_mask], mean, sd)

            alpha = dual_krr_coefficients(x_train, y_train, length, beta)
            y_pred = rbf_kernel(x_val, x_train, length) @ alpha
            score = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
            participant_fold_scores[participant].append(score)

    participant_scores = [
        float(np.mean(scores)) for scores in participant_fold_scores if scores
    ]
    if len(participant_scores) != len(x_list):
        raise RuntimeError("At least one participant lacked a usable validation block.")
    return float(np.mean(participant_scores)), participant_scores


def fit_final_summary(
    x_list: list[np.ndarray],
    raw_y_list: list[np.ndarray],
    length: float,
    beta: float,
) -> dict[str, object]:
    """Fit all standardized ratings and summarize the downstream analysis."""
    standardized = [
        standardize(y, float(y.mean()), float(y.std(ddof=0))) for y in raw_y_list
    ]
    model = RKHSFunctionPCA(
        x_list, standardized, {"length": length, "beta": beta}, modelDim=3
    ).fit()
    contributions = 100 * model.eVal / model.eVal.sum()
    grid = np.linspace(-1.0, 1.0, 401)
    return {
        "contribution_percent_first3": contributions[:3].tolist(),
        "cumulative_percent_first2": float(contributions[:2].sum()),
        "cumulative_percent_first3": float(contributions[:3].sum()),
        "clusters": ward_partition(model.H, cluster_count=4),
        "curves": model.predict(grid),
        "eigenvectors_first3": model.eVec[:, :3],
        "pc_functions_first3": model.pc_functions(grid).T,
    }


def ward_partition(gram: np.ndarray, cluster_count: int) -> list[list[int]]:
    """Ward partition using exact RKHS within-cluster sum-of-squares increases."""
    clusters: list[list[int]] = [[index] for index in range(len(gram))]

    def merge_cost(left: list[int], right: list[int]) -> float:
        left_idx = np.asarray(left, dtype=int)
        right_idx = np.asarray(right, dtype=int)
        left_size = len(left)
        right_size = len(right)
        left_norm = float(gram[np.ix_(left_idx, left_idx)].sum()) / left_size**2
        right_norm = float(gram[np.ix_(right_idx, right_idx)].sum()) / right_size**2
        cross = float(gram[np.ix_(left_idx, right_idx)].sum()) / (
            left_size * right_size
        )
        mean_distance_sq = max(left_norm + right_norm - 2.0 * cross, 0.0)
        return left_size * right_size / (left_size + right_size) * mean_distance_sq

    while len(clusters) > cluster_count:
        best_pair: tuple[int, int] | None = None
        best_cost = np.inf
        for left in range(len(clusters) - 1):
            for right in range(left + 1, len(clusters)):
                cost = merge_cost(clusters[left], clusters[right])
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (left, right)
        if best_pair is None:
            raise RuntimeError("Ward clustering did not find a merge.")
        left, right = best_pair
        merged = sorted(clusters[left] + clusters[right])
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in best_pair
        ]
        clusters.append(merged)

    participant_groups = [[index + 1 for index in group] for group in clusters]
    return sorted(participant_groups, key=lambda group: min(group))


def downstream_comparison(
    participant_full: dict[str, object], training_fold: dict[str, object]
) -> dict[str, object]:
    full_vectors = np.asarray(participant_full["eigenvectors_first3"])
    fold_vectors = np.asarray(training_fold["eigenvectors_first3"])
    singular_values = np.linalg.svd(full_vectors.T @ fold_vectors, compute_uv=False)

    full_pc = np.asarray(participant_full["pc_functions_first3"])
    fold_pc = np.asarray(training_fold["pc_functions_first3"])
    pc_correlations = [
        float(abs(np.corrcoef(full_pc[:, component], fold_pc[:, component])[0, 1]))
        for component in range(3)
    ]

    full_curves = np.asarray(participant_full["curves"])
    fold_curves = np.asarray(training_fold["curves"])
    participant_curve_rmse = np.sqrt(np.mean((full_curves - fold_curves) ** 2, axis=1))

    full_partition = sorted(
        [tuple(group) for group in participant_full["clusters"]]
    )
    fold_partition = sorted(
        [tuple(group) for group in training_fold["clusters"]]
    )
    return {
        "top3_participant_score_subspace_mean_cos2": float(
            np.mean(singular_values**2)
        ),
        "absolute_pc_function_correlations": pc_correlations,
        "mean_participant_curve_rmse_on_z_scale": float(
            participant_curve_rmse.mean()
        ),
        "max_participant_curve_rmse_on_z_scale": float(
            participant_curve_rmse.max()
        ),
        "identical_four_cluster_partition": full_partition == fold_partition,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_list, raw_y_list = load_raw_data()
    validation_masks = make_shared_x_block_masks(x_list)
    combinations = list(product(LENGTH_GRID, BETA_GRID))

    rows: list[dict[str, float | str]] = []
    summaries: dict[str, dict[str, float | str]] = {}
    for scheme in STANDARDIZATION_SCHEMES:
        scheme_rows: list[dict[str, float | str]] = []
        for index, (length, beta) in enumerate(combinations, start=1):
            mean_score, participant_scores = evaluate_params(
                x_list,
                raw_y_list,
                validation_masks,
                float(length),
                float(beta),
                scheme,
            )
            row: dict[str, float | str] = {
                "standardization": scheme,
                "length": float(length),
                "beta": float(beta),
                "mean_rmse": mean_score,
            }
            row.update(
                {
                    f"participant_{i + 1}_rmse": score
                    for i, score in enumerate(participant_scores)
                }
            )
            scheme_rows.append(row)
            print(
                f"{scheme:16s} {index:3d}/{len(combinations)}  "
                f"length={length:.6f}  beta={beta:.6f}  RMSE={mean_score:.6f}",
                flush=True,
            )

        ranked = sorted(scheme_rows, key=lambda row: float(row["mean_rmse"]))
        for rank, row in enumerate(ranked, start=1):
            row["rank_within_scheme"] = rank
        rows.extend(ranked)
        best = ranked[0]
        summaries[scheme] = {
            "standardization": scheme,
            "sd_definition": "population SD (ddof=0)",
            "length": float(best["length"]),
            "beta": float(best["beta"]),
            "mean_rmse": float(best["mean_rmse"]),
        }

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "standardization_sensitivity_cv_results.csv", index=False
    )
    final_summaries = {
        scheme: fit_final_summary(
            x_list,
            raw_y_list,
            float(summary["length"]),
            float(summary["beta"]),
        )
        for scheme, summary in summaries.items()
    }
    downstream = downstream_comparison(
        final_summaries["participant_full"], final_summaries["training_fold"]
    )
    serializable_final = {
        scheme: {
            key: value
            for key, value in summary.items()
            if key
            not in {
                "curves",
                "eigenvectors_first3",
                "pc_functions_first3",
            }
        }
        for scheme, summary in final_summaries.items()
    }
    comparison = {
        "cv": "5-fold contiguous x blocks shared across participants",
        "estimation": "participant-specific dual KRR; no additional jitter; direct cross-kernel RKHS inner products",
        "schemes": summaries,
        "downstream_results_after_full_data_fit": serializable_final,
        "downstream_comparison": downstream,
        "interpretation": (
            "participant_full uses all 50 ratings for each participant's mean and SD; "
            "training_fold uses only training ratings and applies those values to both "
            "training and held-out ratings"
        ),
    }
    (RESULTS_DIR / "standardization_sensitivity_summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
