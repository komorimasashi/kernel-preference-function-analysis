from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(__file__).resolve().parent
MODEL_PATH = PACKAGE_DIR / "results" / "final_model.pkl"
OUTPUT_DIR = PACKAGE_DIR / "figures"
OUTPUT_TAG = "corrected_5fold"
N_CLUSTERS = 4

CLUSTER_COLORS = {
    1: "#E69F00",
    2: "#009E73",
    3: "#D55E00",
    4: "#CC79A7",
}

sys.path.insert(0, str(SOURCE_DIR))

mpl.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def load_model():
    model = joblib.load(MODEL_PATH)
    if model.Z is None:
        model.fit()
    return model


def rkhs_linkage(model):
    norms = np.diag(model.H)
    dist_sq = norms[:, None] + norms[None, :] - 2 * model.H
    dist = np.sqrt(np.maximum(dist_sq, 0.0))
    np.fill_diagonal(dist, 0.0)
    linked = linkage(squareform(dist, checks=False), method="ward")
    return dist, linked


def figure3(model, x_grid, curves):
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    colors = plt.cm.rainbow(np.linspace(0, 1, curves.shape[0]))
    for index, curve in enumerate(curves):
        ax.plot(
            x_grid[:, 0],
            curve,
            color=colors[index],
            linewidth=1.15,
            alpha=0.78,
            label=f"participant {index + 1}",
        )
    ax.set_xlim(-1, 1)
    margin = 0.06 * (curves.max() - curves.min())
    ax.set_ylim(curves.min() - margin, curves.max() + margin)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Standardized preference")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.subplots_adjust(right=0.78)
    fig.savefig(OUTPUT_DIR / "empirical_individual_functions.pdf")
    plt.close(fig)


def cluster_solution(model):
    _, linked = rkhs_linkage(model)
    raw_labels = fcluster(linked, t=N_CLUSTERS, criterion="maxclust")
    leaf_order = np.asarray(dendrogram(linked, no_plot=True)["leaves"], dtype=int)

    ordered_raw = []
    for idx in leaf_order:
        label = int(raw_labels[idx])
        if label not in ordered_raw:
            ordered_raw.append(label)
    relabel = {raw: new for new, raw in enumerate(ordered_raw, start=1)}
    cluster_labels = np.array([relabel[int(value)] for value in raw_labels])
    return linked, cluster_labels, leaf_order


def add_nonoverlapping_participant_labels(ax, x_values, y_values):
    """Place participant IDs near points without letting the labels collide."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    point_pixels = ax.transData.transform(np.column_stack([x_values, y_values]))
    axes_box = ax.get_window_extent(renderer=renderer)
    local_density = np.sum(
        np.linalg.norm(point_pixels[:, None, :] - point_pixels[None, :, :], axis=2) < 48,
        axis=1,
    )
    placement_order = np.argsort(-local_density)
    candidates = [
        (0, 9), (8, 8), (-8, 8), (10, 0), (-10, 0),
        (9, -9), (-9, -9), (0, -11), (14, 12), (-14, 12),
        (16, 0), (-16, 0), (14, -14), (-14, -14), (0, 18), (0, -19),
        (22, 15), (-22, 15), (22, -15), (-22, -15),
    ]
    placed_boxes = []

    for index in placement_order:
        best = None
        best_score = np.inf
        for dx, dy in candidates:
            annotation = ax.annotate(
                str(index + 1),
                xy=(x_values[index], y_values[index]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.12},
            )
            fig.canvas.draw()
            box = annotation.get_window_extent(renderer=renderer).expanded(1.10, 1.18)
            overlap = sum(
                max(0.0, min(box.x1, old.x1) - max(box.x0, old.x0))
                * max(0.0, min(box.y1, old.y1) - max(box.y0, old.y0))
                for old in placed_boxes
            )
            covered_points = sum(
                box.contains(px, py) for px, py in point_pixels if not (px == point_pixels[index, 0] and py == point_pixels[index, 1])
            )
            outside = (
                max(0.0, axes_box.x0 - box.x0)
                + max(0.0, box.x1 - axes_box.x1)
                + max(0.0, axes_box.y0 - box.y0)
                + max(0.0, box.y1 - axes_box.y1)
            )
            score = 1000.0 * overlap + 500.0 * covered_points + 1000.0 * outside + dx * dx + dy * dy
            if score < best_score:
                best_score = score
                best = (dx, dy, box)
            annotation.remove()

        dx, dy, box = best
        arrowprops = None
        if np.hypot(dx, dy) >= 13:
            arrowprops = {"arrowstyle": "-", "color": "#777777", "linewidth": 0.45, "shrinkA": 1, "shrinkB": 3}
        ax.annotate(
            str(index + 1),
            xy=(x_values[index], y_values[index]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.12},
            arrowprops=arrowprops,
        )
        placed_boxes.append(box)


def figure4(model, cluster_labels):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9))
    colors = [CLUSTER_COLORS[int(cluster)] for cluster in cluster_labels]
    pairs = [(0, 1), (0, 2)]
    limit = max(2.0, float(np.max(np.abs(model.Z[:, :3]))) * 1.08)
    for ax, (x_idx, y_idx) in zip(axes, pairs):
        ax.scatter(model.Z[:, x_idx], model.Z[:, y_idx], c=colors, s=20, alpha=0.9)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"PC{x_idx + 1}")
        ax.set_ylabel(f"PC{y_idx + 1}")
        add_nonoverlapping_participant_labels(ax, model.Z[:, x_idx], model.Z[:, y_idx])
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLUSTER_COLORS[cluster],
            markeredgecolor="none",
            markersize=6,
            label=f"Cluster {cluster}",
        )
        for cluster in range(1, N_CLUSTERS + 1)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=N_CLUSTERS,
        frameon=False,
    )
    fig.subplots_adjust(wspace=0.28, bottom=0.16)
    fig.savefig(OUTPUT_DIR / "empirical_pc_scores.pdf")
    plt.close(fig)


def figure5(model, x_grid):
    z_mean = np.mean(model.Z, axis=0)
    z_sd = np.std(model.Z, axis=0)
    generated = {}
    for pc in range(3):
        for multiplier in (-2, 0, 2):
            z = z_mean.copy()
            z[pc] += multiplier * z_sd[pc]
            generated[(pc, multiplier)] = model.generate_pos(x_grid, z[None, :])[0]

    fig, axes = plt.subplots(3, 3, figsize=(8.3, 8.3), sharex=True, sharey=True)
    column_titles = ["-2SD", "Mean", "+2SD"]
    multipliers = [-2, 0, 2]
    all_values = np.concatenate(list(generated.values()))
    y_limit = max(2.2, float(np.max(np.abs(all_values))) * 1.08)
    for row in range(3):
        for col, multiplier in enumerate(multipliers):
            ax = axes[row, col]
            ax.plot(x_grid[:, 0], generated[(row, multiplier)], color="black", linewidth=1.0)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-y_limit, y_limit)
            ax.set_yticks([-2, -1, 0, 1, 2])
            ax.set_xlabel(r"$x$")
            if col == 0:
                ax.set_ylabel("Standardized preference")
            if row == 0:
                ax.set_title(column_titles[col], fontsize=16)
        axes[row, 0].text(
            -0.34,
            0.5,
            f"PC{row + 1}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=16,
        )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.94, bottom=0.08, wspace=0.22, hspace=0.28)
    fig.savefig(OUTPUT_DIR / "empirical_pc_functions.pdf")
    plt.close(fig)


def dendrogram_link_color(model, linked, cluster_labels):
    threshold = linked[-(N_CLUSTERS - 1), 2] + 1e-9
    node_leaves = {idx: {idx} for idx in range(model.task_size)}
    for row_idx, row in enumerate(linked):
        left, right = int(row[0]), int(row[1])
        node_leaves[model.task_size + row_idx] = node_leaves[left] | node_leaves[right]

    def link_color(node_id):
        labels = {int(cluster_labels[idx]) for idx in node_leaves[int(node_id)]}
        if len(labels) == 1:
            return CLUSTER_COLORS[labels.pop()]
        return "#808080"

    return threshold, link_color


def figure6a(model, linked, cluster_labels):
    fig, ax_d = plt.subplots(figsize=(8.0, 3.8))
    threshold, link_color = dendrogram_link_color(model, linked, cluster_labels)

    dendro = dendrogram(
        linked,
        ax=ax_d,
        labels=np.arange(1, model.task_size + 1),
        color_threshold=0,
        above_threshold_color="#808080",
        link_color_func=link_color,
    )
    ax_d.set_xlabel("Participant ID")
    ax_d.set_ylabel("Ward linkage height")

    leaves = np.asarray(dendro["leaves"], dtype=int)
    for cluster in range(1, N_CLUSTERS + 1):
        positions = [5 + 10 * i for i, subject_idx in enumerate(leaves) if cluster_labels[subject_idx] == cluster]
        if positions:
            ax_d.text(
                float(np.mean(positions)),
                threshold * 1.03,
                f"Cluster {cluster}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.20)
    fig.savefig(OUTPUT_DIR / "empirical_cluster_dendrogram.pdf")
    plt.close(fig)

    return leaves


def figure6b(x_grid, curves, cluster_labels):
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.4), sharex=True, sharey=True)
    mini_axes = axes.ravel()
    overall = np.mean(curves, axis=0)
    series = [("Overall Mean", overall)]
    for cluster in range(1, N_CLUSTERS + 1):
        members = np.where(cluster_labels == cluster)[0]
        series.append((f"Cluster {cluster} Mean", np.mean(curves[members], axis=0)))

    mean_values = np.concatenate([values for _, values in series])
    mean_limit = max(1.5, float(np.max(np.abs(mean_values))) * 1.12)
    for series_idx, (ax, (title, values)) in enumerate(zip(mini_axes, series)):
        color = "black" if series_idx == 0 else CLUSTER_COLORS[series_idx]
        ax.plot(x_grid[:, 0], values, color=color, linewidth=1.25)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-mean_limit, mean_limit)
        ax.set_xlabel(r"$x$", fontsize=8)
        ax.set_ylabel("Standardized preference", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in mini_axes[len(series):]:
        ax.axis("off")

    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.10, hspace=0.40, wspace=0.30)
    fig.savefig(OUTPUT_DIR / "empirical_cluster_mean_functions.pdf")
    plt.close(fig)


def figure6(model, x_grid, curves, linked, cluster_labels):
    leaves = figure6a(model, linked, cluster_labels)
    figure6b(x_grid, curves, cluster_labels)
    return cluster_labels, leaves


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    x_grid = np.linspace(-1, 1, 401).reshape(-1, 1)
    curves = model.predict(x_grid)

    linked, cluster_labels, _ = cluster_solution(model)

    figure3(model, x_grid, curves)
    figure4(model, cluster_labels)
    figure5(model, x_grid)
    cluster_labels, leaves = figure6(model, x_grid, curves, linked, cluster_labels)

    pd.DataFrame(
        {
            "participant": np.arange(1, model.task_size + 1),
            "cluster": cluster_labels,
        }
    ).to_csv(PACKAGE_DIR / "results" / "cluster_membership.csv", index=False)

    ratios = 100 * model.eVal / np.sum(model.eVal)
    print(f"params={model.params}")
    print(f"contribution_first3={ratios[:3]}")
    print(f"cumulative_first3={ratios[:3].sum():.8f}")
    print(f"dendrogram_leaf_order={(leaves + 1).tolist()}")
    for cluster in range(1, N_CLUSTERS + 1):
        members = (np.where(cluster_labels == cluster)[0] + 1).tolist()
        print(f"cluster_{cluster}={members}")


def regenerate_cluster_colored_figures():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    x_grid = np.linspace(-1, 1, 401).reshape(-1, 1)
    curves = model.predict(x_grid)
    linked, cluster_labels, _ = cluster_solution(model)
    figure4(model, cluster_labels)
    figure6(model, x_grid, curves, linked, cluster_labels)


if __name__ == "__main__":
    main()
