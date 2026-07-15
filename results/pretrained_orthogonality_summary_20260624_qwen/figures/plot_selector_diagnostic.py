from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parent / "pretrained_orthogonality_pythia70m_fair_benchmark_extended_split_4mods_arc_hella100_lora5_20260627_v3"
BENCHMARK = RUN / "metrics" / "fair_benchmark.csv"
CORRELATIONS = RUN / "metrics" / "correlations.csv"
OUT_DIR = ROOT / "figures"


LABELS = {
    "qsr_rotated_wanda_whitened": "fixed QSR",
    "rqs_rotated_wanda_whitened": "fixed RQS",
    "hessian_guided_qsr_budget": "H-guided QSR",
    "slim_like_srq_proxy": "SLiM-like proxy",
    "spq_like_rsq_no_lora": "SPQ-like",
    "hessian_guided_spq_no_lora": "H-guided SPQ",
    "spq_like_rsq_lora": "SPQ-like + LoRA",
    "hessian_guided_spq_lora": "H-guided SPQ + LoRA",
}


COLORS = {
    "fixed": "#4c78a8",
    "guided": "#9467bd",
    "proxy": "#8c564b",
    "spq": "#17becf",
}


def load_benchmark() -> dict[str, dict[str, str]]:
    with BENCHMARK.open(newline="") as handle:
        return {row["strategy"]: row for row in csv.DictReader(handle)}


def load_correlations() -> dict[str, float]:
    wanted = {
        "additivity": "|rho_H| vs add.",
        "real_ppl": "|rho_H| vs PPL",
        "taylor": "Taylor vs loss",
        "frobenius_baseline": "Frob. vs loss",
        "trace_only_baseline": "Trace vs loss",
    }
    values: dict[str, float] = {}
    with CORRELATIONS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["family"]
            if key in wanted:
                values[wanted[key]] = float(row["spearman_rho"])
    return values


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    rows = load_benchmark()
    correlations = load_correlations()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    qsr_names = [
        "hessian_guided_qsr_budget",
        "rqs_rotated_wanda_whitened",
        "qsr_rotated_wanda_whitened",
        "slim_like_srq_proxy",
    ]
    qsr_rows = [rows[name] for name in qsr_names]
    ax = axes[0]
    qsr_offsets = {
        "qsr_rotated_wanda_whitened": (5, 5, "left"),
        "rqs_rotated_wanda_whitened": (5, 5, "left"),
        "hessian_guided_qsr_budget": (-74, -2, "right"),
        "slim_like_srq_proxy": (-74, 5, "right"),
    }
    for row in qsr_rows:
        name = row["strategy"]
        family = "guided" if "hessian" in name else "proxy" if "slim" in name else "fixed"
        ax.scatter(
            f(row, "predicted_hessian_cost"),
            f(row, "signed_ppl_delta_percent"),
            s=95,
            color=COLORS[family],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        dx, dy, ha = qsr_offsets[name]
        ax.annotate(
            LABELS[name],
            (f(row, "predicted_hessian_cost"), f(row, "signed_ppl_delta_percent")),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha=ha,
        )
    ax.invert_xaxis()
    ax.set_xlabel("Predicted Hessian cost, lower is better")
    ax.set_ylabel("Signed PPL delta (%)")
    ax.set_ylim(57.0, 68.4)
    ax.grid(alpha=0.25)
    ax.text(0.40, 0.07, "Same memory 0.133", transform=ax.transAxes, va="bottom", fontsize=9)

    spq_names = [
        "spq_like_rsq_no_lora",
        "hessian_guided_spq_no_lora",
        "spq_like_rsq_lora",
        "hessian_guided_spq_lora",
    ]
    spq_rows = [rows[name] for name in spq_names]
    ax = axes[1]
    x = list(range(len(spq_rows)))
    ppl = [f(row, "signed_ppl_delta_percent") for row in spq_rows]
    acc = [f(row, "zero_shot_accuracy_delta") for row in spq_rows]
    colors = [COLORS["guided"] if "hessian" in row["strategy"] else COLORS["spq"] for row in spq_rows]
    ax.bar([i - 0.18 for i in x], ppl, width=0.36, color=colors, label="PPL delta")
    ax.set_ylabel("Signed PPL delta (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[row["strategy"]] for row in spq_rows], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.bar([i + 0.18 for i in x], acc, width=0.36, color="#59a14f", alpha=0.78, label="Accuracy delta")
    ax2.axhline(0, color="#59a14f", linewidth=0.8, alpha=0.6)
    ax2.set_ylabel("Mean zero-shot accuracy delta")
    ax.text(0.02, 0.96, "SPQ prior, memory 0.196", transform=ax.transAxes, va="top", fontsize=9)
    ax = axes[2]
    corr_labels = list(correlations.keys())
    corr_values = [correlations[key] for key in corr_labels]
    corr_colors = [
        COLORS["guided"] if "rho_H" in label or "Taylor" in label else "#7f7f7f"
        for label in corr_labels
    ]
    ax.bar(range(len(corr_labels)), corr_values, color=corr_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel("Spearman rho")
    ax.set_xticks(range(len(corr_labels)))
    ax.set_xticklabels(corr_labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.96, "Proxy validity on split run", transform=ax.transAxes, va="top", fontsize=9)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["fixed"], markeredgecolor="black", markersize=8, label="fixed recipe"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["guided"], markeredgecolor="black", markersize=8, label="Hessian-guided"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["spq"], markeredgecolor="black", markersize=8, label="SPQ-like"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["proxy"], markeredgecolor="black", markersize=8, label="SLiM-like proxy"),
        plt.Line2D([0], [0], color="#777777", linewidth=7, label="SPQ panel: left bars = PPL"),
        plt.Line2D([0], [0], color="#59a14f", linewidth=7, label="SPQ panel: right bars = accuracy"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "selector_failure_diagnostic.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / "selector_failure_diagnostic.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
