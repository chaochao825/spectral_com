#!/usr/bin/env python3
"""Plot verified physical-rate and incremental-repair trade-offs.

The figure is generated only from the committed serialized Pythia-70M result.
It therefore describes six selected MLP tensors and one seed, not a full-model
or cross-model result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "pretrained_hessian_repair_pythia70m_serialized_20260714"
ENDPOINTS = RESULT_DIR / "strategy_endpoints.csv"
PAIRS = RESULT_DIR / "paired_method_comparisons.csv"
OUTPUT_STEM = FIGURE_DIR / "exact_rate_tradeoffs"

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#222222",
    "gray": "#777777",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "nan", "NaN"}:
        raise ValueError(f"missing finite {key!r} in row {row}")
    return float(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    endpoint_rows = {
        row["strategy"]: row
        for row in _rows(ENDPOINTS)
        if row.get("artifact_file_bytes") and row.get("target_ratio") == "0.258"
    }
    requested = [
        ("Q", "Q", "o", PALETTE["gray"]),
        ("Q_block_scale", "Q + block scale", "s", PALETTE["sky"]),
        ("Q+S", "Q+S", "^", PALETTE["orange"]),
        ("Q+S_OBS", "Q+S (OBS)", "v", PALETTE["green"]),
        ("Q+L", "Q+L", "*", PALETTE["blue"]),
        (
            "Q+S+L_QL_budget_component_scale",
            "Q+S+L, strict",
            "D",
            PALETTE["vermillion"],
        ),
        ("Q+S+L_component_scale", "Q+S+L, +5,056 B", "P", PALETTE["purple"]),
    ]
    missing = [strategy for strategy, *_ in requested if strategy not in endpoint_rows]
    if missing:
        raise SystemExit(f"missing endpoint rows: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.72), constrained_layout=True)
    ax = axes[0]
    plotted: list[tuple[float, float]] = []
    legend_handles: list[Line2D] = []
    for strategy, label, marker, color in requested:
        row = endpoint_rows[strategy]
        x = _float(row, "artifact_file_bytes") / 1024.0
        y = _float(row, "perplexity_delta")
        plotted.append((x, y))
        face = "none" if "strict" in label else color
        size = 78 if marker == "*" else 43
        ax.scatter(
            [x],
            [y],
            marker=marker,
            s=size,
            facecolors=face,
            edgecolors=color,
            linewidths=1.25,
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=face,
                markeredgecolor=color,
                markeredgewidth=1.15,
                markersize=7,
                label=label,
            )
        )

    # Draw the observed nondominated physical-byte frontier.
    frontier: list[tuple[float, float]] = []
    best_y = float("inf")
    for point in sorted(set(plotted)):
        if point[1] < best_y:
            frontier.append(point)
            best_y = point[1]
    ax.plot(
        [p[0] for p in frontier],
        [p[1] for p in frontier],
        linestyle="--",
        linewidth=1.0,
        color=PALETTE["black"],
        alpha=0.65,
        zorder=2,
    )
    ax.set_xlabel("Serialized artifact (KiB)")
    ax.set_ylabel(r"Perplexity increase $\Delta$PPL")
    ax.grid(axis="both", linestyle=":", linewidth=0.55, alpha=0.45)
    ax.legend(handles=legend_handles, loc="upper right", ncol=1, handletextpad=0.4)
    ax.text(-0.14, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")

    pair_rows = {row["comparison_id"]: row for row in _rows(PAIRS)}
    pair_specs = [
        ("obs_vs_qs", "OBS vs. Q+S", "o", PALETTE["green"]),
        ("block_scale_vs_q", "Block scale vs. Q", "s", PALETTE["sky"]),
        (
            "constrained_scaled_qsl_vs_ql",
            "Strict QSL vs. Q+L",
            "D",
            PALETTE["vermillion"],
        ),
        (
            "unconstrained_scaled_qsl_vs_ql",
            "QSL (+5,056 B) vs. Q+L",
            "P",
            PALETTE["purple"],
        ),
    ]
    missing_pairs = [key for key, *_ in pair_specs if key not in pair_rows]
    if missing_pairs:
        raise SystemExit(f"missing paired rows: {missing_pairs}")

    ax = axes[1]
    for key, label, marker, color in pair_specs:
        row = pair_rows[key]
        x = _float(row, "artifact_file_byte_difference") / 1024.0
        # CSV convention is left-minus-right; invert it so positive means repair benefit.
        y = -_float(row, "paired_mean_nll_difference")
        low = -_float(row, "normal_95_ci_high")
        high = -_float(row, "normal_95_ci_low")
        ax.errorbar(
            [x],
            [y],
            yerr=[[y - low], [high - y]],
            fmt=marker,
            color=color,
            markerfacecolor="none" if "Strict" in label else color,
            markeredgewidth=1.2,
            markersize=6.0,
            capsize=2.5,
            elinewidth=1.0,
            label=label,
            zorder=4,
        )
    ax.axhline(0.0, color=PALETTE["black"], linewidth=0.8, alpha=0.75)
    ax.axvline(0.0, color=PALETTE["gray"], linewidth=0.65, alpha=0.55)
    ax.set_xlabel("Extra serialized bytes over control (KiB)")
    ax.set_ylabel("Paired NLL recovery (positive is better)")
    ax.grid(axis="both", linestyle=":", linewidth=0.55, alpha=0.45)
    ax.legend(loc="upper right", handletextpad=0.4)
    ax.text(-0.14, 1.03, "(b)", transform=ax.transAxes, fontweight="bold")

    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "scope": "Pythia-70M; selected six MLP linear tensors; one seed",
        "evidence_status": "verified_local_experiment",
        "inputs": {
            str(ENDPOINTS.relative_to(ROOT)).replace("\\", "/"): _sha256(ENDPOINTS),
            str(PAIRS.relative_to(ROOT)).replace("\\", "/"): _sha256(PAIRS),
        },
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
