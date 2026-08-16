from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
MODEL_CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_DIR / "results"

sys.path.insert(0, str(MODEL_CODE_DIR))
from model.KPCA5 import SparseKPCA  # noqa: E402
import load_psycho_data2  # noqa: E402


N_FOLDS = 5
JITTER = 1e-6
MODEL_DIM = 3
LENGTH_GRID = np.logspace(-1, 0, 21)
NOISE_GRID = np.logspace(-2, 0, 11)


def rbf(x1: np.ndarray, x2: np.ndarray, length: float) -> np.ndarray:
    dist2 = (x1[:, None] - x2[None, :]) ** 2
    return np.exp(-dist2 / (2.0 * length**2))


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    sd = values.std()
    if sd <= np.finfo(np.float64).eps:
        return np.zeros_like(values)
    return (values - values.mean()) / sd


def make_shared_x_block_masks(x_list: list[np.ndarray]) -> list[list[np.ndarray]]:
    """Split the sorted union of x into five contiguous blocks shared by all participants."""
    x_union = np.unique(np.concatenate([x.reshape(-1) for x in x_list]))
    union_blocks = np.array_split(x_union, N_FOLDS)
    return [
        [np.isin(x.reshape(-1), block) for x in x_list]
        for block in union_blocks
    ]


def evaluate_params(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    validation_masks: list[list[np.ndarray]],
    length: float,
    noise_level: float,
) -> tuple[float, list[float]]:
    """Return mean participant zRMSE and the 20 participant-level means."""
    participant_fold_scores: list[list[float]] = [[] for _ in x_list]
    for fold_masks in validation_masks:
        for participant, (x, y, val_mask) in enumerate(zip(x_list, y_list, fold_masks)):
            train_mask = ~val_mask
            if not np.any(train_mask) or not np.any(val_mask):
                continue
            x_train = x[train_mask, 0]
            x_val = x[val_mask, 0]
            y_train = y[train_mask]
            y_val = y[val_mask]

            # Dual KRR is equivalent to the full-basis KRR step in the zero-jitter
            # limit and avoids constructing the large union basis during CV.
            k_train = rbf(x_train, x_train, length)
            alpha = np.linalg.solve(
                k_train + (noise_level + JITTER) * np.eye(k_train.shape[0]),
                y_train,
            )
            y_pred = rbf(x_val, x_train, length) @ alpha
            score = float(np.sqrt(np.mean((zscore(y_val) - zscore(y_pred)) ** 2)))
            participant_fold_scores[participant].append(score)

    participant_scores = [float(np.mean(scores)) for scores in participant_fold_scores if scores]
    if len(participant_scores) != len(x_list):
        raise RuntimeError("At least one participant was not evaluated in all usable x blocks.")
    return float(np.mean(participant_scores)), participant_scores


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_list, y_list = load_psycho_data2.load_data()
    validation_masks = make_shared_x_block_masks(x_list)

    rows: list[dict[str, float]] = []
    combinations = list(product(LENGTH_GRID, NOISE_GRID))
    for index, (length, noise_level) in enumerate(combinations, start=1):
        mean_score, participant_scores = evaluate_params(
            x_list, y_list, validation_masks, float(length), float(noise_level)
        )
        row: dict[str, float] = {
            "length": float(length),
            "noise_level": float(noise_level),
            "mean_standardized_rmse": mean_score,
        }
        row.update(
            {f"participant_{i + 1}_standardized_rmse": score for i, score in enumerate(participant_scores)}
        )
        rows.append(row)
        print(
            f"{index:3d}/{len(combinations)}  length={length:.6f}  "
            f"noise={noise_level:.6f}  zRMSE={mean_score:.6f}",
            flush=True,
        )

    cv_results = pd.DataFrame(rows).sort_values("mean_standardized_rmse").reset_index(drop=True)
    cv_results.insert(0, "rank", np.arange(1, len(cv_results) + 1))
    cv_results.to_csv(RESULTS_DIR / "xblock_cv_results.csv", index=False)

    best = cv_results.iloc[0]
    selected = {
        "cv": "5-fold contiguous blocks of the sorted union of all participants' x values",
        "preprocessing": "ratings z-standardized separately within each participant",
        "aggregation": "zRMSE within participant and fold, then equal-weight mean across folds and participants",
        "length": float(best["length"]),
        "noise_level": float(best["noise_level"]),
        "beta_precision_in_code": float(1.0 / best["noise_level"]),
        "beta_regularization_for_mean_squared_loss_n50": float(best["noise_level"] / 50.0),
        "mean_standardized_rmse": float(best["mean_standardized_rmse"]),
        "jitter": JITTER,
    }
    (RESULTS_DIR / "selected_parameters.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    params = {"length": selected["length"], "noise_level": selected["noise_level"]}
    model = SparseKPCA(x_list=x_list, y_list=y_list, params=params, modelDim=MODEL_DIM, jitter=JITTER)
    model.fit()
    joblib.dump(model, RESULTS_DIR / "final_model.pkl")

    contribution = 100.0 * model.eVal / model.eVal.sum()
    pd.DataFrame(
        {
            "component": np.arange(1, contribution.size + 1),
            "contribution_percent": contribution,
        }
    ).to_csv(RESULTS_DIR / "contribution_ratios.csv", index=False)
    print("selected_parameters=", selected)
    print("first_three_contributions=", contribution[:3])
    print("first_three_cumulative=", contribution[:3].sum())


if __name__ == "__main__":
    main()
