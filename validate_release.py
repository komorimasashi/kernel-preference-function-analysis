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
    assert np.isclose(selected["length"], 0.31622776601683794)
    assert np.isclose(selected["beta"], 0.0031697863849222286)
    assert np.isclose(selected["mean_standardized_rmse"], 0.9205714160486792, rtol=0, atol=1e-10)
    assert "training-rating mean and population SD" in selected["preprocessing"]
    assert "participant-specific dual KRR" in selected["estimation"]
    assert "no additional jitter" in selected["estimation"]

    contributions = pd.read_csv(result_dir / "contribution_ratios.csv")
    assert np.allclose(
        contributions.loc[:2, "contribution_percent"],
        [45.3079102792, 32.8073291067, 9.8821389520],
        rtol=0, atol=1e-9,
    )
    assert np.isclose(contributions.loc[:1, "contribution_percent"].sum(), 78.1152393859, rtol=0, atol=1e-9)
    assert np.isclose(contributions.loc[:2, "contribution_percent"].sum(), 87.9973783379, rtol=0, atol=1e-9)

    membership = pd.read_csv(result_dir / "cluster_membership.csv")
    counts = membership["cluster"].value_counts().sort_index().to_dict()
    assert counts == {1: 5, 2: 4, 3: 7, 4: 4}, counts
    expected_members = {
        1: [4, 7, 10, 11, 15], 2: [12, 13, 14, 20],
        3: [1, 3, 5, 6, 8, 9, 18], 4: [2, 16, 17, 19],
    }
    for label, members in expected_members.items():
        assert sorted(membership.loc[membership.cluster == label, "participant"]) == members

    sensitivity = json.loads(
        (result_dir / "standardization_sensitivity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    full = sensitivity["schemes"]["participant_full"]
    fold = sensitivity["schemes"]["training_fold"]
    assert np.isclose(full["length"], 0.28183829312644537)
    assert np.isclose(full["beta"], 0.005023772863019165)
    assert np.isclose(fold["length"], selected["length"])
    assert np.isclose(fold["beta"], selected["beta"])
    downstream = sensitivity["downstream_comparison"]
    assert downstream["top3_participant_score_subspace_mean_cos2"] > 0.998
    assert downstream["identical_four_cluster_partition"] is True

    # Both entry points must produce the same primary CV table and final fit.
    main_cv = pd.read_csv(result_dir / "xblock_cv_results.csv").sort_values(["length", "beta"])
    sensitivity_cv = pd.read_csv(result_dir / "standardization_sensitivity_cv_results.csv")
    fold_cv = sensitivity_cv[sensitivity_cv.standardization == "training_fold"].sort_values(["length", "beta"])
    assert len(main_cv) == len(fold_cv) == 231
    assert np.allclose(main_cv[["length", "beta"]], fold_cv[["length", "beta"]], rtol=0, atol=1e-14)
    assert np.allclose(main_cv.mean_standardized_rmse, fold_cv.mean_rmse, rtol=0, atol=1e-12)
    for participant in range(1, 21):
        assert np.allclose(
            main_cv[f"participant_{participant}_standardized_rmse"],
            fold_cv[f"participant_{participant}_rmse"], rtol=0, atol=1e-12,
        )
    best = main_cv.sort_values("mean_standardized_rmse").iloc[0]
    assert np.isclose(best.length, selected["length"], rtol=0, atol=1e-14)
    assert np.isclose(best.beta, selected["beta"], rtol=0, atol=1e-14)
    assert np.isclose(best.mean_standardized_rmse, selected["mean_standardized_rmse"], rtol=0, atol=1e-12)
    sensitivity_fit = sensitivity["downstream_results_after_full_data_fit"]["training_fold"]
    assert np.allclose(
        sensitivity_fit["contribution_percent_first3"],
        contributions.loc[:2, "contribution_percent"], rtol=0, atol=1e-10,
    )


def validate_simulation_results() -> None:
    simulation_dir = ROOT / "simulation" / "results"
    shape = pd.read_csv(
        simulation_dir / "simulation1_metric_ablation" / "metric_ablation_summary.csv"
    )
    assert set(shape["metric"]) == {"zRMSE"}
    expected_zrmse = {
        (3, "Raw"): 0.5592393819,
        (3, "L2"): 0.4748123084,
        (3, "RKHS"): 0.6461880651,
        (5, "Raw"): 0.4836557048,
        (5, "L2"): 0.3249345083,
        (5, "RKHS"): 0.3864208808,
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
        (0.5, 1): 0.1840567542,
        (0.5, 2): 0.0849062982,
        (0.5, 3): 0.0458542276,
        (1.0, 1): 0.3099940481,
        (1.0, 2): 0.1628732350,
        (1.0, 3): 0.0541546565,
        (2.0, 1): 0.2379297326,
        (2.0, 2): 0.0814994854,
        (2.0, 3): -0.0861592273,
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

    main_regularization = pd.read_csv(
        simulation_dir
        / "simulation2_sensitivity"
        / "regularization_selection_summary.csv"
    )
    empirical_regularization = pd.read_csv(
        simulation_dir
        / "simulation2_T20_N50"
        / "regularization_selection_summary.csv"
    )
    expected_upper_boundary = {
        (15, 0.5): 0.005,
        (30, 0.5): 0.0,
        (50, 0.5): 0.0,
    }
    for (observation_count, snr), expected in expected_upper_boundary.items():
        row = main_regularization[
            (main_regularization["N_obs"] == observation_count)
            & (main_regularization["SNR"] == snr)
        ]
        assert len(row) == 1
        assert np.isclose(
            row.iloc[0]["proportion_at_upper_grid_boundary"], expected
        )
    row = empirical_regularization[empirical_regularization["SNR"] == 0.5]
    assert len(row) == 1
    assert np.isclose(row.iloc[0]["proportion_at_upper_grid_boundary"], 0.0)

    reference_dir = simulation_dir / "simulation2_sensitivity"
    oracle_reference = pd.read_csv(
        reference_dir / "oracle_l2_reference_replication_results.csv"
    )
    oracle_reference_summary = pd.read_csv(
        reference_dir / "oracle_l2_reference_summary.csv"
    )
    assert len(oracle_reference) == 9 * 200 * 3
    assert len(oracle_reference_summary) == 3
    assert set(oracle_reference_summary["L"]) == {1, 2, 3}
    assert (oracle_reference_summary["n_reps"] == 9 * 200).all()
    assert not oracle_reference.duplicated(
        subset=["N_obs", "SNR", "L", "rep"]
    ).any()
    assert oracle_reference["oracle_l2_to_rkhs_mean_cos2"].between(0, 1).all()
    pooled = (
        oracle_reference.groupby("L")["oracle_l2_to_rkhs_mean_cos2"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values("L")
    )
    pooled["ci_low"] = pooled["mean"] - 1.96 * pooled["std"] / np.sqrt(
        pooled["count"]
    )
    pooled["ci_high"] = pooled["mean"] + 1.96 * pooled["std"] / np.sqrt(
        pooled["count"]
    )
    saved = oracle_reference_summary.sort_values("L")
    assert np.array_equal(saved["n_reps"], pooled["count"])
    for column in ("mean", "ci_low", "ci_high"):
        assert np.allclose(saved[column], pooled[column], atol=1e-12)

    validate_simulation_aggregates(simulation_dir)


def validate_simulation_aggregates(simulation_dir: Path) -> None:
    """Recompute reported means, intervals and paired differences from replications."""
    def check_interval(row, values, mean_column="mean"):
        mean = values.mean()
        half_width = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
        assert len(values) == 200
        assert np.allclose(
            [row[mean_column], row["ci_low"], row["ci_high"]],
            [mean, mean - half_width, mean + half_width], rtol=0, atol=1e-12,
        )

    shape_dir = simulation_dir / "simulation1_metric_ablation"
    shape_reps = pd.read_csv(shape_dir / "metric_ablation_replication_results.csv")
    shape_summary = pd.read_csv(shape_dir / "metric_ablation_summary.csv")
    assert len(shape_reps) == 200 * 5
    assert not shape_reps.duplicated(["rep", "L"]).any()
    for _, row in shape_summary.iterrows():
        values = shape_reps.loc[
            shape_reps["L"] == row["L"], f'{row["condition"]}_zRMSE_mean'
        ]
        check_interval(row, values)

    for directory, condition_count in [("simulation2_sensitivity", 9), ("simulation2_T20_N50", 3)]:
        result_dir = simulation_dir / directory
        reps = pd.read_csv(result_dir / "sensitivity_replication_results.csv")
        summary = pd.read_csv(result_dir / "sensitivity_summary.csv")
        contrasts = pd.read_csv(result_dir / "sensitivity_paired_contrasts.csv")
        assert len(reps) == condition_count * 200 * 3
        assert not reps.duplicated(["N_obs", "SNR", "rep", "L"]).any()
        for frame, paired in [(summary, False), (contrasts, True)]:
            for _, row in frame.iterrows():
                subset = reps[
                    (reps["N_obs"] == row["N_obs"])
                    & (reps["SNR"] == row["SNR"])
                    & (reps["L"] == row["L"])
                ]
                prefix = row["target"]
                if paired:
                    values = (subset[f"{prefix}_RKHS_function_mean_cos2"]
                              - subset[f"{prefix}_L2_function_mean_cos2"])
                    check_interval(row, values, "rkhs_advantage")
                else:
                    values = subset[f'{prefix}_{row["method"]}_function_mean_cos2']
                    assert values.between(0, 1).all()
                    check_interval(row, values)


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
