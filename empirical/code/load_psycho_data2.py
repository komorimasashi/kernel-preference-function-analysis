from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"


def load_raw_data() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load predictor values and unstandardized ratings for 20 participants."""
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for participant in range(1, 21):
        values = pd.read_csv(DATA_DIR / f"data{participant}.csv").to_numpy()
        x = values[:, 2].astype(np.float64).reshape(-1, 1)
        y = values[:, 1].astype(np.float64)
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list


def population_zscore(values: np.ndarray) -> np.ndarray:
    """Standardize with the population SD (denominator N; NumPy ddof=0)."""
    y_sd = values.std(ddof=0)
    if y_sd == 0:
        raise ValueError("Cannot standardize ratings with zero variance.")
    return (values - values.mean()) / y_sd


def load_data() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load ratings and z-standardize all observations within participant."""
    x_list, raw_y_list = load_raw_data()
    y_list: list[np.ndarray] = []
    for participant, y_raw in enumerate(raw_y_list, start=1):
        y_sd = y_raw.std(ddof=0)
        if y_sd == 0:
            raise ValueError(f"Participant {participant} has zero rating variance.")
        y_list.append(population_zscore(y_raw))

    return x_list, y_list
