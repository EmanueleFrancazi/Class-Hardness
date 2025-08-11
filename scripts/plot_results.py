"""Visualize simulation metrics for different class subsets.

Specify the simulation settings in `SIMULATION_SETTINGS` below. Each setting
is a dictionary with two keys:

- ``path``: directory containing ``run_*`` subfolders with ``metrics.npz`` files
- ``classes``: list of class labels used in the simulation

Example::

    SIMULATION_SETTINGS = [
        {"path": "results/mnist_0_1_2", "classes": [0, 1, 2]},
        {"path": "results/mnist_3_4_5", "classes": [3, 4, 5]},
    ]

Run the script with ``python scripts/plot_results.py``. The script automatically
loads all available runs for each setting and produces two figures: one with
per-class loss (top row) and representation overlaps (bottom row), and another
replacing the loss with per-class accuracy. It also generates a scatter plot
comparing dynamic and static overlaps across runs.
"""

from __future__ import annotations

import glob
import os
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Use Times New Roman for all text
plt.rcParams["font.family"] = "Times New Roman"

# ---------------------------------------------------------------------------
# User configuration: specify result directories and class lists here
# ---------------------------------------------------------------------------
SIMULATION_SETTINGS: List[Dict] = [
    # {"path": "results/mnist_0_1_2", "classes": [0, 1, 2]},
    # {"path": "results/mnist_3_4_5", "classes": [3, 4, 5]},
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_runs(setting: Dict) -> List[Dict[str, np.ndarray]]:
    """Load ``metrics.npz`` for all runs under ``setting['path']``.

    Missing or corrupt files are skipped.
    """
    pattern = os.path.join(setting["path"], "run_*", "metrics.npz")
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        try:
            data = np.load(f)
            runs.append({k: data[k] for k in data.files})
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Could not load {f}: {exc}")
    return runs


def aggregate_loss(runs: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return steps, mean and standard error of per-class loss."""
    steps = runs[0]["steps"]
    loss = np.stack([r["loss_per_class"] for r in runs])  # (runs, steps, classes)
    mean = loss.mean(axis=0)
    se = loss.std(axis=0, ddof=1) / np.sqrt(len(runs))
    return steps, mean, se


def aggregate_accuracy(runs: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return steps, mean and standard error of per-class accuracy."""
    steps = runs[0]["steps"]
    acc = np.stack([r["accuracy_per_class"] for r in runs])  # (runs, steps, classes)
    mean = acc.mean(axis=0)
    se = acc.std(axis=0, ddof=1) / np.sqrt(len(runs))
    return steps, mean, se


def aggregate_overlap(runs: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """Return steps and mean/SE overlap for each class pair."""
    steps = runs[0]["steps"]
    overlaps = np.stack([r["overlap_dynamic"] for r in runs])  # (runs, steps, classes, classes)
    num_classes = overlaps.shape[-1]
    pair_means: Dict[Tuple[int, int], np.ndarray] = {}
    pair_ses: Dict[Tuple[int, int], np.ndarray] = {}
    for i, j in combinations(range(num_classes), 2):
        vals = overlaps[:, :, i, j]
        pair_means[(i, j)] = vals.mean(axis=0)
        pair_ses[(i, j)] = vals.std(axis=0, ddof=1) / np.sqrt(len(runs))
    return steps, pair_means, pair_ses


def collect_scatter_points(setting: Dict, runs: List[Dict[str, np.ndarray]]):
    """Gather dynamic/static overlap pairs for scatter plot."""
    classes = setting["classes"]
    pairs = list(combinations(range(len(classes)), 2))
    points = []
    for r in runs:
        dyn = r["overlap_dynamic"][-1]  # final step
        stat = r["overlap_static"]
        for i, j in pairs:
            points.append({
                "setting": ",".join(map(str, classes)),
                "path": setting["path"],
                "pair": f"{classes[i]}-{classes[j]}",
                "dyn": dyn[i, j],
                "stat": stat[i, j],
            })
    return points


# ---------------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------------

def main():
    # Load all runs for each setting
    loaded = []
    scatter_points = []
    for setting in SIMULATION_SETTINGS:
        runs = load_runs(setting)
        if not runs:
            print(f"No runs found for setting {setting}")
            continue
        loaded.append((setting, runs))
        scatter_points.extend(collect_scatter_points(setting, runs))

    if not loaded:
        print("No data loaded; check SIMULATION_SETTINGS paths.")
        return

    num_settings = len(loaded)
    fig_loss, axes_loss = plt.subplots(2, num_settings, figsize=(6 * num_settings, 8), sharex='col')
    fig_acc, axes_acc = plt.subplots(2, num_settings, figsize=(6 * num_settings, 8), sharex='col')

    for col, (setting, runs) in enumerate(loaded):
        classes = setting["classes"]

        steps_loss, mean_loss, se_loss = aggregate_loss(runs)
        steps_acc, mean_acc, se_acc = aggregate_accuracy(runs)
        steps_ov, mean_ov, se_ov = aggregate_overlap(runs)

        x_loss = steps_loss + 1
        x_acc = steps_acc + 1
        x_ov = steps_ov + 1

        ax_top_loss = axes_loss[0, col]
        for idx, cls in enumerate(classes):
            ax_top_loss.plot(x_loss, mean_loss[:, idx], label=f"Class {cls}")
            ax_top_loss.fill_between(x_loss, mean_loss[:, idx] - se_loss[:, idx], mean_loss[:, idx] + se_loss[:, idx], alpha=0.3)
        ax_top_loss.set_xscale('log')
        ax_top_loss.set_ylabel('Per-class loss')
        ax_top_loss.set_title(f"Classes: {','.join(map(str, classes))}")
        ax_top_loss.legend()

        ax_top_acc = axes_acc[0, col]
        for idx, cls in enumerate(classes):
            ax_top_acc.plot(x_acc, mean_acc[:, idx], label=f"Class {cls}")
            ax_top_acc.fill_between(x_acc, mean_acc[:, idx] - se_acc[:, idx], mean_acc[:, idx] + se_acc[:, idx], alpha=0.3)
        ax_top_acc.set_xscale('log')
        ax_top_acc.set_ylabel('Per-class accuracy')
        ax_top_acc.set_title(f"Classes: {','.join(map(str, classes))}")
        ax_top_acc.legend()

        ax_bottom_loss = axes_loss[1, col]
        ax_bottom_acc = axes_acc[1, col]
        for (i, j), mean in mean_ov.items():
            label = f"{classes[i]}-{classes[j]}"
            ax_bottom_loss.plot(x_ov, mean, label=label)
            ax_bottom_loss.fill_between(x_ov, mean - se_ov[(i, j)], mean + se_ov[(i, j)], alpha=0.3)
            ax_bottom_acc.plot(x_ov, mean, label=label)
            ax_bottom_acc.fill_between(x_ov, mean - se_ov[(i, j)], mean + se_ov[(i, j)], alpha=0.3)
        ax_bottom_loss.set_xscale('log')
        ax_bottom_loss.set_ylabel('Representation overlap')
        ax_bottom_loss.set_xlabel('Training step')
        ax_bottom_loss.legend()

        ax_bottom_acc.set_xscale('log')
        ax_bottom_acc.set_ylabel('Representation overlap')
        ax_bottom_acc.set_xlabel('Training step')
        ax_bottom_acc.legend()

    fig_loss.tight_layout()
    fig_acc.tight_layout()

    # Scatter plot of dynamic vs static overlaps
    if scatter_points:
        fig_scatter, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.tab10.colors
        markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
        setting_to_color = {s["path"]: colors[i % len(colors)] for i, (s, _) in enumerate(loaded)}
        pair_to_marker: Dict[str, str] = {}
        next_marker = 0
        for p in scatter_points:
            key = p["pair"]
            if key not in pair_to_marker:
                pair_to_marker[key] = markers[next_marker % len(markers)]
                next_marker += 1
            ax.scatter(
                p["dyn"],
                p["stat"],
                color=setting_to_color[p["path"]],
                marker=pair_to_marker[key],
                label=f"{p['setting']} | {key}",
            )
        ax.set_xlabel('Dynamic overlap')
        ax.set_ylabel('Static overlap')
        ax.set_title('Dynamic vs. Static overlap')
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicates in legend
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize='small', frameon=False)
        fig_scatter.tight_layout()
    else:
        print("No scatter points to plot.")

    plt.show()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
