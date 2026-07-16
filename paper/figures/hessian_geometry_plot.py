#!/usr/bin/env python3
"""Plot the verified aggregate Hessian correlation matrix for strict Q/S/L."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "results"
    / "pretrained_hessian_repair_pythia70m_serialized_20260714"
    / "strategy_endpoints.csv"
)
OUTPUT_STEM = FIGURE_DIR / "hessian_geometry"
TARGET = "Q+S+L_QL_budget_component_scale"


def main() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("strategy") == TARGET and row.get("target_ratio") == "0.258"
        ]
    if len(matches) != 1:
        raise SystemExit(f"expected one strict Q/S/L row, found {len(matches)}")
    row = matches[0]
    rho_qs = float(row["rho_qs"])
    rho_ql = float(row["rho_ql"])
    rho_sl = float(row["rho_sl"])
    matrix = np.array(
        [[1.0, rho_qs, rho_ql], [rho_qs, 1.0, rho_sl], [rho_ql, rho_sl, 1.0]],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(3.05, 2.55), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
    labels = ["Q", "S", "L"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Perturbation component")
    ax.set_ylabel("Perturbation component")
    for i in range(3):
        for j in range(3):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.52 else "#111111",
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.052, pad=0.04)
    colorbar.set_label(r"$H$-metric correlation $\rho_H$", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)
    manifest = {
        "schema_version": 1,
        "scope": "Pythia-70M; conservative Q/S/L endpoint tail-padded to Q+L final bytes; natural underfill 15,680 bytes; six selected MLP tensors; one seed",
        "evidence_status": "verified_local_experiment",
        "strategy": TARGET,
        "input": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "rho_qs": rho_qs,
        "rho_ql": rho_ql,
        "rho_sl": rho_sl,
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
