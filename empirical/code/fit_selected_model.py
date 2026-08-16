"""Fit the empirical model using the already selected CV parameters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_DIR / "results"

sys.path.insert(0, str(CODE_DIR))
import load_psycho_data2  # noqa: E402
from model.KPCA5 import SparseKPCA  # noqa: E402


MODEL_DIM = 3


def main() -> None:
    selected = json.loads(
        (RESULTS_DIR / "selected_parameters.json").read_text(encoding="utf-8")
    )
    x_list, y_list = load_psycho_data2.load_data()
    params = {
        "length": float(selected["length"]),
        "noise_level": float(selected["noise_level"]),
    }
    model = SparseKPCA(
        x_list=x_list,
        y_list=y_list,
        params=params,
        modelDim=MODEL_DIM,
        jitter=float(selected["jitter"]),
    )
    model.fit()
    joblib.dump(model, RESULTS_DIR / "final_model.pkl")

    contribution = 100.0 * model.eVal / model.eVal.sum()
    pd.DataFrame(
        {
            "component": np.arange(1, contribution.size + 1),
            "contribution_percent": contribution,
        }
    ).to_csv(RESULTS_DIR / "contribution_ratios.csv", index=False)
    print("selected_parameters=", params)
    print("first_three_contributions=", contribution[:3])
    print("first_three_cumulative=", contribution[:3].sum())


if __name__ == "__main__":
    main()
