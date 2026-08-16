"""Create the simulation figures used in the manuscript.

The script reads only the final, expanded-grid ablation and the final latent
subspace sensitivity analyses.  It deliberately does not use the earlier
narrow-grid or pilot results.
"""

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
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save_figure(fig, stem)


def plot_advantage_heatmaps(
    contrasts: pd.DataFrame,
    metric: str,
    color_limit: float,
    stem: str,
) -> None:
    data = contrasts[
        (contrasts["target"] == "OracleRKHS")
        & (contrasts["metric"] == metric)
    ].copy()
    n_values = [15, 30, 50]
    snr_values = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(
        1, 3, figsize=(10.6, 3.35), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for ax, retained in zip(axes, [1, 2, 3]):
        values = np.full((len(snr_values), len(n_values)), np.nan)
        lows = np.full_like(values, np.nan)
        highs = np.full_like(values, np.nan)
        for row_index, snr in enumerate(snr_values):
            for column_index, n_obs in enumerate(n_values):
                row = data[
                    (data["L"] == retained)
                    & (data["SNR"] == snr)
                    & (data["N_obs"] == n_obs)
                ].iloc[0]
                values[row_index, column_index] = row["rkhs_advantage"]
                lows[row_index, column_index] = row["ci_low"]
                highs[row_index, column_index] = row["ci_high"]
        image = ax.imshow(
            values,
            origin="lower",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            aspect="equal",
        )
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                significant = (lows[row_index, column_index] > 0) or (
                    highs[row_index, column_index] < 0
                )
                label = f"{values[row_index, column_index]:+.3f}"
                if significant:
                    label += "*"
                color = "white" if abs(values[row_index, column_index]) > 0.6 * color_limit else "black"
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=color,
                )
        ax.set_title(f"$L={retained}$")
        ax.set_xticks(range(len(n_values)), labels=n_values)
        ax.set_yticks(range(len(snr_values)), labels=["0.5", "1", "2"])
        ax.set_xlabel("Observations per participant, $N$")
    axes[0].set_ylabel("SNR")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("RKHS-PCA advantage in mean squared cosine")
    save_figure(fig, stem)


def main() -> None:
    shape_summary = pd.read_csv(SHAPE_SUMMARY)
    contrasts = pd.read_csv(SUBSPACE_CONTRASTS)
    plot_reconstruction_metric(
        shape_summary,
        metric="zRMSE",
        ylabel="z-standardized RMSE",
        stem="simulation1_shape_zrmse",
    )
    plot_reconstruction_metric(
        shape_summary,
        metric="dX_peak",
        ylabel="Peak localization error",
        stem="simulation1_peak_error",
    )
    plot_advantage_heatmaps(
        contrasts,
        metric="function_mean_cos2",
        color_limit=0.35,
        stem="simulation2_function_subspace",
    )
    plot_advantage_heatmaps(
        contrasts,
        metric="score_mean_cos2",
        color_limit=0.45,
        stem="simulation2_score_subspace",
    )


if __name__ == "__main__":
    main()
