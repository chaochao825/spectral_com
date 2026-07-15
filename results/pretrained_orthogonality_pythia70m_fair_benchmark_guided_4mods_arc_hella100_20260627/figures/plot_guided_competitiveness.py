from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "metrics" / "fair_benchmark.csv"


def short_name(name: str) -> str:
    mapping = {
        "q_only_rtn_4bit": "Q RTN",
        "q_only_rotated_4bit": "Q rot",
        "s_only_magnitude_keep0p8": "S mag",
        "s_only_wanda_keep0p8": "S Wanda",
        "r_only_svd_rank0p5": "R SVD",
        "r_only_whitened_rank0p5": "R white",
        "qsr_naive_rtn_magnitude_svd": "QSR naive",
        "qsr_rotated_wanda_whitened": "QSR rot",
        "rqs_rotated_wanda_whitened": "RQS rot",
        "hessian_guided_qsr_budget": "Hessian",
    }
    return mapping.get(name, name)


def family_color(family: str) -> str:
    return {
        "q_only": "#4c78a8",
        "s_only": "#59a14f",
        "r_only": "#f28e2b",
        "qsr_stack": "#e15759",
        "hessian_guided_stack": "#7b2cbf",
    }.get(family, "#7f7f7f")


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["strategy"] != "baseline"]


def main() -> None:
    rows = load_rows()
    labels = [short_name(row["strategy"]) for row in rows]
    colors = [family_color(row["family"]) for row in rows]
    ppl = [float(row["signed_ppl_delta_percent"]) for row in rows]
    acc = [float(row["zero_shot_accuracy_delta"]) for row in rows]
    memory = [float(row["nominal_memory_ratio"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    ax = axes[0, 0]
    cap = 30.0
    clipped = [max(-cap, min(value, cap)) for value in ppl]
    ax.bar(range(len(rows)), clipped, color=colors)
    for idx, value in enumerate(ppl):
        if value > cap:
            ax.text(idx, cap + 0.8, f"{value:.0f}%", ha="center", va="bottom", fontsize=8, rotation=90)
        elif value < -cap:
            ax.text(idx, -cap - 0.8, f"{value:.0f}%", ha="center", va="top", fontsize=8, rotation=90)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Signed PPL delta (%)")
    lower = min(-2.0, min(clipped) - 4.0)
    upper = max(4.0, max(clipped) + 4.0)
    ax.set_ylim(lower, upper)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.text(0.01, 0.94, "All methods, capped at 30%", transform=ax.transAxes, fontsize=10)

    ax = axes[0, 1]
    ax.bar(range(len(rows)), acc, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean zero-shot accuracy delta")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")

    ax = axes[1, 0]
    scatter_offsets = {
        "qsr_naive_rtn_magnitude_svd": (8, 6),
        "qsr_rotated_wanda_whitened": (8, -10),
        "rqs_rotated_wanda_whitened": (8, 6),
        "hessian_guided_qsr_budget": (8, 16),
        "r_only_svd_rank0p5": (8, 2),
    }
    for row, x, y in zip(rows, memory, ppl):
        ax.scatter(x, y, s=80, color=family_color(row["family"]), edgecolor="black", linewidth=0.4)
        offset = scatter_offsets.get(row["strategy"], (5, 2))
        ax.annotate(short_name(row["strategy"]), (x, y), xytext=offset, textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=5)
    ax.set_xlabel("Nominal memory ratio (lower is smaller)")
    ax.set_ylabel("Signed PPL delta (%)")
    ax.grid(True, which="both", alpha=0.25)

    qsr_names = {
        "qsr_rotated_wanda_whitened",
        "rqs_rotated_wanda_whitened",
        "hessian_guided_qsr_budget",
    }
    qsr = [row for row in rows if row["strategy"] in qsr_names]
    qsr_labels = [short_name(row["strategy"]) for row in qsr]
    qsr_ppl = [float(row["signed_ppl_delta_percent"]) for row in qsr]
    qsr_acc = [float(row["zero_shot_accuracy_delta"]) for row in qsr]
    x = range(len(qsr))
    ax = axes[1, 1]
    ax.bar([i - 0.18 for i in x], qsr_ppl, width=0.36, color="#e15759", label="PPL delta (%)")
    ax2 = ax.twinx()
    ax2.bar([i + 0.18 for i in x], qsr_acc, width=0.36, color="#4c78a8", label="Accuracy delta")
    ax.set_ylabel("Signed PPL delta (%)")
    ax2.set_ylabel("Mean zero-shot accuracy delta")
    ax.set_xticks(list(x))
    ax.set_xticklabels(qsr_labels, rotation=20, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(0, color="#4c78a8", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, max(qsr_ppl) * 1.18)
    ax2.set_ylim(min(qsr_acc) * 1.8, max(0.01, max(qsr_acc) * 1.5))
    ax.text(0.01, 0.94, "Same 0.133 nominal-memory QSR budget", transform=ax.transAxes, fontsize=10)

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=family_color(fam), markersize=9, label=label)
        for fam, label in [
            ("q_only", "Q-only"),
            ("s_only", "S-only"),
            ("r_only", "R-only"),
            ("qsr_stack", "fixed QSR"),
            ("hessian_guided_stack", "Hessian-guided"),
        ]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ROOT / "figures" / "fair_benchmark_guided_competitiveness.png", dpi=220, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fair_benchmark_guided_competitiveness.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
