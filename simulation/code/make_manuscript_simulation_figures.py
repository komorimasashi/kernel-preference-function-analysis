"""Create the two simulation figures used in the current manuscript."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIMULATION_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = SIMULATION_DIR / "figures"
SHAPE_SUMMARY = (
    SIMULATION_DIR
    / "results"
    / "simulation1_metric_ablation"
    / "metric_ablation_summary.csv"
)
SUBSPACE_CONTRASTS = (
    SIMULATION_DIR
    / "results"
    / "simulation2_sensitivity"
    / "sensitivity_paired_contrasts.csv"
)
SUBSPACE_SUMMARY = (
    SIMULATION_DIR
    / "results"
    / "simulation2_sensitivity"
    / "sensitivity_summary.csv"
)
SUBSPACE_ORACLE_REFERENCE = (
    SIMULATION_DIR
    / "results"
    / "simulation2_sensitivity"
    / "oracle_l2_reference_summary.csv"
)


METHODS = {
    "Raw": {"label": "Raw-PCA", "color": "#5B5B5B", "marker": "o"},
    "L2": {"label": "KRR-L2-PCA", "color": "#0072B2", "marker": "s"},
    "RKHS": {"label": "KRR-RKHS-PCA", "color": "#D55E00", "marker": "^"},
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for condition, style in METHODS.items():
        data = summary[
            (summary["condition"] == condition) & (summary["metric"] == metric)
        ].sort_values("L")
        yerr = np.vstack(
            [data["mean"] - data["ci_low"], data["ci_high"] - data["mean"]]
        )
        ax.errorbar(
            data["L"],
            data["mean"],
            yerr=yerr,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=5.5,
            capsize=3,
        )
    ax.set_xlabel("Number of retained PCs, $L$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.30, 1.01)
    ax.set_yticks(np.arange(0.3, 1.01, 0.1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save_figure(fig, stem)


def plot_subspace_accuracy(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    oracle_reference: pd.DataFrame,
    metric: str,
    stem: str,
) -> None:
    accuracy = summary[
        (summary["target"] == "OracleRKHS")
        & (summary["metric"] == metric)
    ].copy()
    differences = contrasts[
        (contrasts["target"] == "OracleRKHS")
        & (contrasts["metric"] == metric)
    ].copy()
    n_values = [15, 30, 50]
    snr_values = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10.6, 7.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row_index, snr in enumerate(snr_values):
        for column_index, retained in enumerate([1, 2, 3]):
            ax = axes[row_index, column_index]
            for method in ("L2", "RKHS"):
                style = METHODS[method]
                data = accuracy[
                    (accuracy["L"] == retained)
                    & (accuracy["SNR"] == snr)
                    & (accuracy["method"] == method)
                ].sort_values("N_obs")
                yerr = np.vstack(
                    [data["mean"] - data["ci_low"], data["ci_high"] - data["mean"]]
                )
                ax.errorbar(
                    data["N_obs"],
                    data["mean"],
                    yerr=yerr,
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.6,
                    markersize=4.8,
                    capsize=2.5,
                )

            reference = oracle_reference[
                oracle_reference["L"] == retained
            ].iloc[0]
            ax.fill_between(
                n_values,
                reference["ci_low"],
                reference["ci_high"],
                color="#6F6F6F",
                alpha=0.16,
                linewidth=0,
            )
            ax.axhline(
                reference["mean"],
                label="Noiseless L2-PCA reference, $c$",
                color="#595959",
                linestyle="--",
                linewidth=1.3,
            )
            ax.axhline(1.0, color="#A0A0A0", linestyle=":", linewidth=1.0)

            paired = differences[
                (differences["L"] == retained)
                & (differences["SNR"] == snr)
            ].sort_values("N_obs")
            for item in paired.itertuples(index=False):
                if item.ci_low > 0 or item.ci_high < 0:
                    ax.text(
                        item.N_obs,
                        0.985,
                        "*",
                        ha="center",
                        va="top",
                        fontsize=10,
                    )

            if row_index == 0:
                ax.set_title(f"$L={retained}$")
            if column_index == 0:
                ax.set_ylabel(f"SNR = {snr:g}\nMean squared cosine")
            if row_index == len(snr_values) - 1:
                ax.set_xlabel("Observations per participant, $N$")
            ax.set_xticks(n_values)
            ax.set_ylim(0.0, 1.03)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    save_figure(fig, stem)


def main() -> None:
    shape_summary = pd.read_csv(SHAPE_SUMMARY)
    subspace_summary = pd.read_csv(SUBSPACE_SUMMARY)
    contrasts = pd.read_csv(SUBSPACE_CONTRASTS)
    oracle_reference = pd.read_csv(SUBSPACE_ORACLE_REFERENCE)
    plot_reconstruction_metric(
        shape_summary,
        metric="zRMSE",
        ylabel="z-standardized RMSE",
        stem="simulation1_shape_zrmse",
    )
    plot_subspace_accuracy(
        subspace_summary,
        contrasts,
        oracle_reference,
        metric="function_mean_cos2",
        stem="simulation2_function_subspace",
    )


if __name__ == "__main__":
    main()
