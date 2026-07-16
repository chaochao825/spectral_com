#!/usr/bin/env python3
"""Plot verified endpoint paths and local Taylor-fit residuals."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "results"
    / "pretrained_hessian_repair_pythia70m_serialized_20260714"
    / "comfort_sweep.csv"
)
OUTPUT_STEM = FIGURE_DIR / "loss_landscape"
PALETTE = ["#777777", "#56B4E9", "#009E73", "#0072B2", "#D55E00"]
STRATEGIES = [
    ("Q", "Q"),
    ("Q_block_scale", "Q + block scale"),
    ("Q+S_OBS", "Q+S (OBS)"),
    ("Q+L", "Q+L"),
    ("Q+S+L_QL_budget_component_scale", "Q+S+L, strict"),
]


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def main() -> None:
    _style()
    grouped: dict[str, list[dict[str, str]]] = {}
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("target_ratio") == "0.258":
                grouped.setdefault(row["strategy"], []).append(row)
    missing = [key for key, _ in STRATEGIES if key not in grouped]
    if missing:
        raise SystemExit(f"missing loss-landscape strategies: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), constrained_layout=True)
    for (key, label), color in zip(STRATEGIES, PALETTE):
        rows = sorted(grouped[key], key=lambda row: float(row["epsilon"]))
        epsilon = [float(row["epsilon"]) for row in rows]
        actual = [float(row["nll_delta"]) for row in rows]
        fitted = [float(row["taylor_fit_nll_delta"]) for row in rows]
        axes[0].plot(
            epsilon,
            actual,
            color=color,
            linewidth=1.45,
            marker="o",
            markersize=2.8,
            markevery=[0, 2, 4, 6, 8, 10, 12],
            label=label,
        )
        axes[1].plot(
            epsilon,
            [a - f for a, f in zip(actual, fitted)],
            color=color,
            linewidth=1.45,
            marker="o",
            markersize=2.8,
            markevery=[0, 2, 4, 6, 8, 10, 12],
            label=label,
        )

    for ax in axes:
        ax.axhline(0.0, color="#222222", linewidth=0.75, alpha=0.7)
        ax.axvline(0.25, color="#999999", linestyle="--", linewidth=0.75)
        ax.set_xlabel(r"Path scale $\epsilon$")
        ax.grid(axis="both", linestyle=":", linewidth=0.55, alpha=0.45)
    axes[0].set_ylabel(r"Held-out NLL change $\Delta\mathcal{L}$")
    axes[1].set_ylabel("NLL minus local Taylor fit")
    axes[0].legend(loc="upper left", ncol=1, handlelength=1.8)
    axes[0].text(-0.14, 1.03, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.14, 1.03, "(b)", transform=axes[1].transAxes, fontweight="bold")

    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)
    manifest = {
        "schema_version": 1,
        "scope": "Pythia-70M; selected six MLP linear tensors; one seed; 13-point paths",
        "evidence_status": "verified_local_experiment",
        "input": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "outputs": [
            OUTPUT_STEM.with_suffix(".pdf").name,
            OUTPUT_STEM.with_suffix(".svg").name,
            OUTPUT_STEM.with_suffix(".png").name,
        ],
    }
    OUTPUT_STEM.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(OUTPUT_STEM.with_suffix(".pdf"))
    print(OUTPUT_STEM.with_suffix(".svg"))
    print(OUTPUT_STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
