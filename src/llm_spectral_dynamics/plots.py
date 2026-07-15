from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_eigenspectrum(eigenvalues: np.ndarray, path: str | Path, *, title: str) -> None:
    plt = _plt()
    vals = np.asarray(eigenvalues, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ranks = np.arange(1, vals.size + 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=160)
    ax.loglog(ranks, vals, linewidth=1.6)
    ax.set_xlabel("rank")
    ax.set_ylabel("covariance eigenvalue")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_eigenspectrum_overlay(
    curves: Iterable[dict[str, object]],
    path: str | Path,
    *,
    title: str,
    normalize: bool = True,
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    plotted = 0
    for curve in curves:
        vals = np.asarray(curve["eigenvalues"], dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue
        if normalize:
            vals = vals / max(float(vals.sum()), 1e-30)
        ranks = np.arange(1, vals.size + 1)
        ax.loglog(ranks, vals, linewidth=1.6, label=str(curve.get("label", f"curve{plotted}")))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("rank")
    ax.set_ylabel("normalized eigenvalue" if normalize else "covariance eigenvalue")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metric_heatmap(
    rows: Iterable[dict[str, object]],
    path: str | Path,
    *,
    metric: str,
    title: str,
) -> None:
    plt = _plt()
    data = [row for row in rows if metric in row and row.get(metric) is not None]
    if not data:
        return
    layers = sorted({int(row["layer"]) for row in data})
    def site_label(row: dict[str, object]) -> str:
        site = str(row["site"])
        head = row.get("head")
        if head is None or head == "":
            return site
        return f"{site}_h{int(head)}"

    sites = sorted({site_label(row) for row in data})
    grid = np.full((len(layers), len(sites)), np.nan, dtype=np.float64)
    layer_index = {layer: i for i, layer in enumerate(layers)}
    site_index = {site: i for i, site in enumerate(sites)}
    for row in data:
        try:
            grid[layer_index[int(row["layer"])], site_index[site_label(row)]] = float(row[metric])
        except (TypeError, ValueError):
            continue
    fig, ax = plt.subplots(figsize=(max(4.0, 0.7 * len(sites)), max(3.0, 0.3 * len(layers))), dpi=160)
    image = ax.imshow(grid, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(np.arange(len(sites)), labels=sites, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(layers)), labels=[str(layer) for layer in layers])
    ax.set_xlabel("site")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
