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
from model.rkhs_pca import RKHSFunctionPCA, dual_krr_coefficients, rbf_kernel  # noqa: E402
import load_psycho_data2  # noqa: E402


N_FOLDS = 5
MODEL_DIM = 3
LENGTH_GRID = np.logspace(-1, 0, 21)
# This is beta in the manuscript objective:
# mean squared error + beta * RKHS norm squared.
# The endpoints correspond to the former dual shifts 0.01 and 1.0 at N=50.
BETA_GRID = np.logspace(np.log10(0.01 / 50.0), np.log10(1.0 / 50.0), 11)


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
    raw_y_list: list[np.ndarray],
    validation_masks: list[list[np.ndarray]],
    length: float,
    beta: float,
) -> tuple[float, list[float]]:
    """Return RMSE after standardizing from each participant's training fold."""
    participant_fold_scores: list[list[float]] = [[] for _ in x_list]
    for fold_masks in validation_masks:
        for participant, (x, y_raw, val_mask) in enumerate(
            zip(x_list, raw_y_list, fold_masks)
        ):
            train_mask = ~val_mask
            if not np.any(train_mask) or not np.any(val_mask):
                continue
            x_train = x[train_mask, 0]
            x_val = x[val_mask, 0]
            train_mean = float(y_raw[train_mask].mean())
            train_sd = float(y_raw[train_mask].std(ddof=0))
            if train_sd == 0.0:
                raise ValueError(
                    f"Participant {participant + 1} has zero training-fold variance."
                )
            y_train = (y_raw[train_mask] - train_mean) / train_sd
            y_val = (y_raw[val_mask] - train_mean) / train_sd

            alpha = dual_krr_coefficients(x_train, y_train, length, beta)
            y_pred = rbf_kernel(x_val, x_train, length) @ alpha
            # The validation responses are transformed only with the training
            # fold's mean and population SD, so no held-out response enters the
            # preprocessing of the fitted model.
            score = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
            participant_fold_scores[participant].append(score)

    participant_scores = [float(np.mean(scores)) for scores in participant_fold_scores if scores]
    if len(participant_scores) != len(x_list):
        raise RuntimeError("At least one participant was not evaluated in all usable x blocks.")
    return float(np.mean(participant_scores)), participant_scores


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_list, raw_y_list = load_psycho_data2.load_raw_data()
    validation_masks = make_shared_x_block_masks(x_list)

    rows: list[dict[str, float]] = []
    combinations = list(product(LENGTH_GRID, BETA_GRID))
    for index, (length, beta) in enumerate(combinations, start=1):
        mean_score, participant_scores = evaluate_params(
            x_list, raw_y_list, validation_masks, float(length), float(beta)
        )
        row: dict[str, float] = {
            "length": float(length),
            "beta": float(beta),
            "mean_standardized_rmse": mean_score,
        }
        row.update(
            {f"participant_{i + 1}_standardized_rmse": score for i, score in enumerate(participant_scores)}
        )
        rows.append(row)
        print(
            f"{index:3d}/{len(combinations)}  length={length:.6f}  "
            f"beta={beta:.6f}  zRMSE={mean_score:.6f}",
            flush=True,
        )

    cv_results = pd.DataFrame(rows).sort_values("mean_standardized_rmse").reset_index(drop=True)
    cv_results.insert(0, "rank", np.arange(1, len(cv_results) + 1))
    cv_results.to_csv(RESULTS_DIR / "xblock_cv_results.csv", index=False)

    best = cv_results.iloc[0]
    selected = {
        "cv": "5-fold contiguous blocks of the sorted union of all participants' x values",
        "preprocessing": "within each participant and fold, the training-rating mean and population SD (ddof=0) are applied to both training and held-out ratings",
        "validation_criterion": "RMSE between held-out ratings and predictions transformed with training-fold statistics",
        "aggregation": "RMSE within participant and fold, then equal-weight mean across folds and participants",
        "length": float(best["length"]),
        "beta": float(best["beta"]),
        "mean_standardized_rmse": float(best["mean_standardized_rmse"]),
        "estimation": "participant-specific dual KRR: K_tt + N_t * beta * I; no additional jitter",
        "rkhs_inner_product": "alpha_t.T @ K_ttprime @ alpha_tprime",
    }
    (RESULTS_DIR / "selected_parameters.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # After selecting the hyperparameters without response leakage, refit the
    # final descriptive model using all observations z-standardized within each
    # participant (population SD, ddof=0).
    _, y_list = load_psycho_data2.load_data()
    params = {
        "length": selected["length"],
        "beta": selected["beta"],
    }
    model = RKHSFunctionPCA(x_list=x_list, y_list=y_list, params=params, modelDim=MODEL_DIM)
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
