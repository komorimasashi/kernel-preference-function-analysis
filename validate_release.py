"""Validate the public data package and manuscript-facing numerical results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def validate_empirical_data() -> None:
    data_dir = ROOT / "empirical" / "data"
    files = sorted(
        data_dir.glob("data*.csv"),
        key=lambda path: int(path.stem.removeprefix("data")),
    )
    assert len(files) == 20, f"Expected 20 participant files, found {len(files)}"
    expected_columns = ["aspect_ratio", "beauty", "random_value"]
    for path in files:
        data = pd.read_csv(path)
        assert list(data.columns) == expected_columns, f"Unexpected columns in {path}"
        assert len(data) == 50, f"Expected 50 trials in {path}, found {len(data)}"
        assert not data.isna().any().any(), f"Missing value in {path}"
        assert data["beauty"].between(1, 7).all(), f"Rating outside 1--7 in {path}"
        assert np.allclose(data["beauty"], np.round(data["beauty"])), (
            f"Non-integer rating in {path}"
        )
        assert data["random_value"].between(-1, 1).all(), (
            f"Predictor outside [-1, 1] in {path}"
        )
        assert np.allclose(
            data["aspect_ratio"], 4.0 ** data["random_value"], rtol=1e-12, atol=1e-12
        ), f"Aspect-ratio transform mismatch in {path}"


def validate_empirical_results() -> None:
    result_dir = ROOT / "empirical" / "results"
    selected = json.loads(
        (result_dir / "selected_parameters.json").read_text(encoding="utf-8")
    )
    assert np.isclose(selected["length"], 0.44668359215096315)
    assert np.isclose(selected["beta_regularization_for_mean_squared_loss_n50"], 0.02)
    assert np.isclose(selected["mean_standardized_rmse"], 1.0844346148368396)

    contributions = pd.read_csv(result_dir / "contribution_ratios.csv")
    assert np.allclose(
        contributions.loc[:2, "contribution_percent"],
        [60.3007649861, 31.0014044439, 5.0934334810],
        atol=1e-9,
    )
    assert np.isclose(contributions.loc[:1, "contribution_percent"].sum(), 91.30216943)
    assert np.isclose(contributions.loc[:2, "contribution_percent"].sum(), 96.39560291)

    membership = pd.read_csv(result_dir / "cluster_membership.csv")
    counts = membership["cluster"].value_counts().sort_index().to_dict()
    assert counts == {1: 3, 2: 6, 3: 4, 4: 5, 5: 2}, counts


def validate_simulation_results() -> None:
    simulation_dir = ROOT / "simulation" / "results"
    shape = pd.read_csv(
        simulation_dir / "simulation1_metric_ablation" / "metric_ablation_summary.csv"
    )
    assert set(shape["metric"]) == {"zRMSE"}
    expected_zrmse = {
        (3, "Raw"): 0.5592393834,
        (3, "L2"): 0.4810738626,
        (3, "RKHS"): 0.6314874029,
        (5, "Raw"): 0.4836557065,
        (5, "L2"): 0.3398056583,
        (5, "RKHS"): 0.4221389693,
    }
    for (retained, condition), expected in expected_zrmse.items():
        row = shape[
            (shape["L"] == retained)
            & (shape["condition"] == condition)
            & (shape["metric"] == "zRMSE")
        ]
        assert len(row) == 1
        assert np.isclose(row.iloc[0]["mean"], expected, atol=1e-10)

    empirical_design = pd.read_csv(
        simulation_dir
        / "simulation2_T20_N50"
        / "sensitivity_paired_contrasts.csv"
    )
    expected_advantage = {
        (0.5, 1): 0.1093275086,
        (0.5, 2): 0.0343445855,
        (0.5, 3): 0.0013921720,
        (1.0, 1): 0.2057051442,
        (1.0, 2): 0.1174323187,
        (1.0, 3): 0.0458937282,
        (2.0, 1): 0.2993997303,
        (2.0, 2): 0.1788148914,
        (2.0, 3): 0.0757858321,
    }
    subset = empirical_design[
        (empirical_design["target"] == "OracleRKHS")
        & (empirical_design["metric"] == "function_mean_cos2")
    ]
    for (snr, retained), expected in expected_advantage.items():
        row = subset[(subset["SNR"] == snr) & (subset["L"] == retained)]
        assert len(row) == 1
        assert np.isclose(row.iloc[0]["rkhs_advantage"], expected, atol=1e-10)

    main_contrasts = pd.read_csv(
        simulation_dir
        / "simulation2_sensitivity"
        / "sensitivity_paired_contrasts.csv"
    )
    assert set(main_contrasts["metric"]) == {"function_mean_cos2"}
    assert set(empirical_design["metric"]) == {"function_mean_cos2"}


def validate_simulation_package() -> None:
    simulation_dir = ROOT / "simulation"
    code_files = {
        path.name for path in (simulation_dir / "code").glob("*.py")
    }
    assert code_files == {
        "make_manuscript_simulation_figures.py",
        "run_latent_subspace_sensitivity.py",
        "run_latent_subspace_sensitivity_T20_N50.py",
        "run_metric_ablation.py",
        "simulation_common.py",
    }
    assert not list((simulation_dir / "code").glob("*.ipynb"))

    figure_files = {
        path.name for path in (simulation_dir / "figures").iterdir() if path.is_file()
    }
    assert figure_files == {
        "simulation1_shape_zrmse.pdf",
        "simulation1_shape_zrmse.png",
        "simulation2_function_subspace.pdf",
        "simulation2_function_subspace.png",
    }


def main() -> None:
    validate_empirical_data()
    validate_empirical_results()
    validate_simulation_results()
    validate_simulation_package()
    print("Release validation passed: data structure and manuscript-facing results match.")


if __name__ == "__main__":
    main()
