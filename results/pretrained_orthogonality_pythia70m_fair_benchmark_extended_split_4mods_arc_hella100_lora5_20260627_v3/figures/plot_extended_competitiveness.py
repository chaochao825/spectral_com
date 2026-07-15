from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "metrics" / "fair_benchmark.csv"


SHORT = {
    "q_only_rtn_4bit": "Q RTN",
    "q_only_rotated_4bit": "Q rot",
    "s_only_magnitude_keep0p8": "S mag",
    "s_only_wanda_keep0p8": "S Wanda",
    "r_only_svd_rank0p5": "R SVD",
    "r_only_whitened_rank0p5": "R white",
    "qsr_naive_rtn_magnitude_svd": "QSR naive",
    "qsr_rotated_wanda_whitened": "QSR rot",
    "rqs_rotated_wanda_whitened": "RQS rot",
    "hessian_guided_qsr_budget": "H-QSR",
    "slim_like_srq_proxy": "SLiM proxy",
    "spq_like_rsq_no_lora": "SPQ",
    "hessian_guided_spq_no_lora": "H-SPQ",
    "spq_like_rsq_lora": "SPQ+LoRA",
    "hessian_guided_spq_lora": "H-SPQ+LoRA",
}


COLORS = {
    "q_only": "#4c78a8",
    "s_only": "#59a14f",
    "r_only": "#f28e2b",
    "qsr_stack": "#e15759",
    "hessian_guided_stack": "#7b2cbf",
    "slim_proxy": "#8c564b",
    "spq_like": "#17becf",
    "hessian_guided_spq": "#9467bd",
}


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["strategy"] != "baseline"]
    return sorted(rows, key=lambda row: float(row["signed_ppl_delta_percent"]))


def label(row: dict[str, str]) -> str:
    return SHORT.get(row["strategy"], row["strategy"])


def color(row: dict[str, str]) -> str:
    return COLORS.get(row["family"], "#7f7f7f")


def main() -> None:
    rows = load_rows()
    labels = [label(row) for row in rows]
    colors = [color(row) for row in rows]
    ppl = [float(row["signed_ppl_delta_percent"]) for row in rows]
    acc = [float(row["zero_shot_accuracy_delta"]) for row in rows]
    mem = [float(row["nominal_memory_ratio"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))

    ax = axes[0, 0]
    cap = 40.0
    clipped = [max(-cap, min(value, cap)) for value in ppl]
    ax.bar(range(len(rows)), clipped, color=colors)
    for idx, value in enumerate(ppl):
        if abs(value) > cap:
            ax.text(idx, cap + 1.0 if value > 0 else -cap - 1.0, f"{value:.0f}%", ha="center", va="bottom" if value > 0 else "top", fontsize=8, rotation=90)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Signed PPL delta (%)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(min(-2.0, min(clipped) - 5.0), max(5.0, max(clipped) + 5.0))
    ax.text(0.01, 0.94, "All fair rows sorted by PPL; capped at 40%", transform=ax.transAxes, fontsize=10)

    ax = axes[0, 1]
    ax.bar(range(len(rows)), acc, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean zero-shot accuracy delta")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")

    ax = axes[1, 0]
    offsets = {
        "qsr_naive_rtn_magnitude_svd": (8, 14),
        "qsr_rotated_wanda_whitened": (8, -10),
        "rqs_rotated_wanda_whitened": (8, 4),
        "hessian_guided_qsr_budget": (8, 18),
        "slim_like_srq_proxy": (8, 30),
        "spq_like_rsq_no_lora": (8, 8),
        "hessian_guided_spq_no_lora": (8, -12),
        "spq_like_rsq_lora": (8, 22),
        "hessian_guided_spq_lora": (8, -26),
        "s_only_magnitude_keep0p8": (8, -10),
        "s_only_wanda_keep0p8": (8, 6),
    }
    for row, x, y in zip(rows, mem, ppl):
        ax.scatter(x, y, s=80, color=color(row), edgecolor="black", linewidth=0.4)
        ax.annotate(label(row), (x, y), xytext=offsets.get(row["strategy"], (5, 3)), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=5)
    ax.set_xlabel("Nominal memory ratio")
    ax.set_ylabel("Signed PPL delta (%)")
    ax.grid(True, which="both", alpha=0.25)

    focus_names = [
        "qsr_rotated_wanda_whitened",
        "rqs_rotated_wanda_whitened",
        "hessian_guided_qsr_budget",
        "slim_like_srq_proxy",
        "spq_like_rsq_no_lora",
        "hessian_guided_spq_no_lora",
        "spq_like_rsq_lora",
        "hessian_guided_spq_lora",
    ]
    by_name = {row["strategy"]: row for row in rows}
    focus = [by_name[name] for name in focus_names if name in by_name]
    x = list(range(len(focus)))
    ax = axes[1, 1]
    ax.bar([i - 0.18 for i in x], [float(row["signed_ppl_delta_percent"]) for row in focus], width=0.36, color=[color(row) for row in focus], label="PPL")
    ax2 = ax.twinx()
    ax2.bar([i + 0.18 for i in x], [float(row["zero_shot_accuracy_delta"]) for row in focus], width=0.36, color="#4c78a8", alpha=0.72, label="Accuracy")
    ax.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(0, color="#4c78a8", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Signed PPL delta (%)")
    ax2.set_ylabel("Mean zero-shot accuracy delta")
    ax.set_xticks(x)
    ax.set_xticklabels([label(row) for row in focus], rotation=30, ha="right")
    ax.text(0.01, 0.94, "Low-memory recipe focus", transform=ax.transAxes, fontsize=10)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=value, markersize=9, label=name)
        for name, value in [
            ("Q-only", COLORS["q_only"]),
            ("S-only", COLORS["s_only"]),
            ("R-only", COLORS["r_only"]),
            ("fixed QSR", COLORS["qsr_stack"]),
            ("H-QSR", COLORS["hessian_guided_stack"]),
            ("SLiM proxy", COLORS["slim_proxy"]),
            ("SPQ-like", COLORS["spq_like"]),
            ("H-SPQ", COLORS["hessian_guided_spq"]),
        ]
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ROOT / "figures" / "fair_benchmark_extended_competitiveness.png", dpi=220, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fair_benchmark_extended_competitiveness.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
