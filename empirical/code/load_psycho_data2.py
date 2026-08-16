from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"


def load_data() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load 20 participants and standardize ratings within each participant."""
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for participant in range(1, 21):
        values = pd.read_csv(DATA_DIR / f"data{participant}.csv").to_numpy()
        x = values[:, 2].astype(np.float64).reshape(-1, 1)
        y_raw = values[:, 1].astype(np.float64)
        y_sd = y_raw.std()
        if y_sd == 0:
            raise ValueError(f"Participant {participant} has zero rating variance.")
        y = (y_raw - y_raw.mean()) / y_sd
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list
