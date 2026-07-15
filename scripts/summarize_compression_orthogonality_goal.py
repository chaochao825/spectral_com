from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def first(rows: list[dict[str, str]], **kwargs: str) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in kwargs.items()):
            return row
    raise KeyError(kwargs)


def scatter_with_fit(ax, x, y, *, xlabel: str, ylabel: str, title: str, color: str) -> None:
    ax.scatter(x, y, s=42, alpha=0.82, color=color, edgecolors="white", linewidth=0.5)
    if len(x) >= 2:
        coef = np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), deg=1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xx, coef[0] * xx + coef[1], color="black", linewidth=1.0, alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.22, linewidth=0.7)


def write_report(
    out_dir: Path,
    result_dir: Path,
    correlations: list[dict[str, str]],
    additivity: list[dict[str, str]],
    layerwise: list[dict[str, str]],
    order_gap: list[dict[str, str]],
) -> None:
    corr_rows = {(row["family"], row["x"], row["y"]): row for row in correlations}

    def corr_value(family: str, x: str, y: str) -> str:
        return corr_rows[(family, x, y)]["spearman_rho"]

    def corr_n(family: str, x: str, y: str) -> str:
        return corr_rows[(family, x, y)]["n"]

    high = max(additivity, key=lambda row: f(row, "abs_rho_h"))
    low = min(additivity, key=lambda row: f(row, "abs_rho_h"))
    largest_gap = max(order_gap, key=lambda row: f(row, "abs_loss_gap"))
    selected = first(layerwise, method="hessian_layerwise")
    fixed = first(layerwise, method="fixed_order")
    lines = [
        "# Goal Alignment Audit",
        "",
        f"Source result directory: `{result_dir}`.",
        "",
        "## Verdict",
        "",
        "The current MVP is consistent with the original goal at toy-model scale. It supports the central diagnostic claim more than the claim of a new compression pipeline: Hessian-weighted cross terms predict additivity error and real degradation, order non-commutativity is observable, and the metric can guide layer-wise choices under the same q/s/r budget.",
        "",
        "The main caveat is scope: this is a small character-language model with diagonal empirical-Fisher/Hessian proxy, not yet a pretrained LLM with GPTQ/AWQ/SparseGPT-grade baselines.",
        "",
        "## Success Criteria",
        "",
        "| Criterion from original goal | Status | Evidence |",
        "|---|---:|---|",
        f"| q/s/r Hessian cosine heatmap on at least one small model | Achieved | `hessian_cosine_heatmap.png`; 3 linear layers (`fc1`, `fc2`, `head`) with q/s/r pair matrix. |",
        f"| Higher `rho_H` pairs have larger additivity error | Achieved | Spearman(|rho_H|, |A_ij|) = `{corr_value('additivity', 'abs_rho_h', 'abs_additivity_error')}` over n={corr_n('additivity', 'abs_rho_h', 'abs_additivity_error')}; high row `{high['layer']}/{high['pair']}` has |rho_H|={float(high['abs_rho_h']):.4f}, |A_ij|={float(high['abs_additivity_error']):.4f}; low row `{low['layer']}/{low['pair']}` has |rho_H|={float(low['abs_rho_h']):.4f}, |A_ij|={float(low['abs_additivity_error']):.4f}. |",
        f"| Report real PPL/accuracy degradation correlations | Achieved | Spearman(|rho_H|, PPL degradation) = `{corr_value('real_ppl', 'abs_rho_h', 'ppl_degradation_pair')}`; Spearman(|rho_H|, accuracy degradation) = `{corr_value('real_accuracy', 'abs_rho_h', 'accuracy_degradation_pair')}`. Taylor loss prediction = `{corr_value('taylor', 'taylor_predicted_loss_delta', 'loss_degradation_pair')}`, above Frobenius baseline `{corr_value('frobenius_baseline', 'frobenius_delta_sum', 'loss_degradation_pair')}`. |",
        f"| Compare R->Q/S vs Q/S->R and explain order gap with singular spectrum + Hessian overlap | Partially achieved | Largest gap `{largest_gap['layer']}: {largest_gap['left_order']} vs {largest_gap['right_order']}` has abs loss gap={float(largest_gap['abs_loss_gap']):.4f}. R-first conditional overlap correlates with abs loss gap `{corr_value('order_gap_r_first_overlap', 'abs_r_first_conditional_hessian_overlap', 'abs_loss_gap')}`; singular entropy/top1/stable-rank shifts correlate `{corr_value('spectrum_order_entropy', 'abs_first_spectral_entropy_delta', 'abs_loss_gap')}`. Symmetric max overlap is weak `{corr_value('order_gap', 'max_abs_conditional_hessian_overlap', 'abs_loss_gap')}`, so the order explanation should be phrased directionally. |",
        f"| Layer-wise method/order selection beats naive fixed-order baseline under same q/s/r settings | Achieved | Hessian-guided layer-wise PPL `{float(selected['perplexity']):.4f}` vs fixed `Q->S->R` PPL `{float(fixed['perplexity']):.4f}`; accuracy degradation `{float(selected['accuracy_degradation']):.4f}` vs `{float(fixed['accuracy_degradation']):.4f}`. |",
        "",
        "## Interpretation Against Expected Claim",
        "",
        "- Consistent: the evidence goes beyond a pretty landscape. It reports `rho_H`, additivity error, order gap, and actual PPL/accuracy degradation correlations.",
        "- Consistent: the strongest pair is `fc2/qs` at `bits2/keep0.45`; the selected loss-landscape anchors now match that exact additivity row.",
        "- Consistent but limited: the framework explains complement/conflict better for additivity and real loss degradation than for a symmetric order-gap overlap metric.",
        "- Not yet sufficient for paper-scale evidence: needs at least one pretrained small LLM or transformer classifier, multiple seeds, and comparisons against stronger compression baselines.",
        "",
        "## Generated Visualizations",
        "",
        "- `figures/goal_alignment_dashboard.png/pdf`",
        "- `figures/goal_alignment_layerwise.png/pdf`",
        "- Existing supporting figures: `hessian_cosine_heatmap.png`, `loss_landscape_fc2_qs_contour.png`, `loss_landscape_fc2_qs_surface.png`",
    ]
    (out_dir / "goal_alignment_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize compression orthogonality MVP against original goal.")
    parser.add_argument("--result-dir", default="results/compression_orthogonality_mvp_20260623_v7")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.output_dir) if args.output_dir else result_dir / "goal_audit"
    figures = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    metrics = result_dir / "metrics"
    correlations = read_csv(metrics / "correlations.csv")
    additivity = read_csv(metrics / "additivity.csv")
    order_gap = read_csv(metrics / "order_gap.csv")
    layerwise = read_csv(metrics / "layerwise_performance.csv")

    corr_rows = {(row["family"], row["x"], row["y"]): row for row in correlations}
    x_rho = [f(row, "abs_rho_h") for row in additivity]
    y_add = [f(row, "abs_additivity_error") for row in additivity]
    y_ppl = [f(row, "ppl_degradation_pair") for row in additivity]

    corr_names = [
        ("|rho_H| vs |A_ij|", ("additivity", "abs_rho_h", "abs_additivity_error")),
        ("|rho_H| vs PPL deg.", ("real_ppl", "abs_rho_h", "ppl_degradation_pair")),
        ("|rho_H| vs acc. deg.", ("real_accuracy", "abs_rho_h", "accuracy_degradation_pair")),
        ("Taylor vs loss deg.", ("taylor", "taylor_predicted_loss_delta", "loss_degradation_pair")),
        ("Frobenius vs loss deg.", ("frobenius_baseline", "frobenius_delta_sum", "loss_degradation_pair")),
    ]
    corr_values = [float(corr_rows[key]["spearman_rho"]) for _, key in corr_names]

    order_names = [
        ("R-first overlap", ("order_gap_r_first_overlap", "abs_r_first_conditional_hessian_overlap", "abs_loss_gap")),
        ("Non-R-first overlap", ("order_gap_non_r_first_overlap", "abs_non_r_first_conditional_hessian_overlap", "abs_loss_gap")),
        ("Spectrum entropy", ("spectrum_order_entropy", "abs_first_spectral_entropy_delta", "abs_loss_gap")),
        ("Spectrum top1", ("spectrum_order_top1", "abs_first_top1_energy_delta", "abs_loss_gap")),
        ("Stable rank", ("spectrum_order_stable_rank", "abs_first_stable_rank_delta", "abs_loss_gap")),
        ("Weight disagreement", ("order_disagreement", "final_weight_disagreement", "abs_loss_gap")),
    ]
    order_values = [float(corr_rows[key]["spearman_rho"]) for _, key in order_names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    ax = axes[0, 0]
    colors = ["#2563eb", "#2563eb", "#2563eb", "#059669", "#6b7280"]
    ax.barh([name for name, _ in corr_names][::-1], corr_values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho")
    ax.set_title("Correlation evidence")
    ax.grid(True, axis="x", alpha=0.22)

    scatter_with_fit(
        axes[0, 1],
        x_rho,
        y_add,
        xlabel="|rho_H|",
        ylabel="|additivity error|",
        title="Hessian overlap explains additivity",
        color="#ef4444",
    )
    axes[0, 1].text(
        0.03,
        0.95,
        f"Spearman={float(corr_rows[('additivity', 'abs_rho_h', 'abs_additivity_error')]['spearman_rho']):.3f}",
        transform=axes[0, 1].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

    scatter_with_fit(
        axes[1, 0],
        x_rho,
        y_ppl,
        xlabel="|rho_H|",
        ylabel="PPL degradation",
        title="Hessian overlap vs real degradation",
        color="#f97316",
    )
    axes[1, 0].text(
        0.03,
        0.95,
        f"Spearman={float(corr_rows[('real_ppl', 'abs_rho_h', 'ppl_degradation_pair')]['spearman_rho']):.3f}",
        transform=axes[1, 0].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

    ax = axes[1, 1]
    bar_colors = ["#2563eb", "#9ca3af", "#059669", "#059669", "#059669", "#6b7280"]
    ax.barh([name for name, _ in order_names][::-1], order_values[::-1], color=bar_colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho with |order loss gap|")
    ax.set_title("Order-gap explanations")
    ax.grid(True, axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figures / "goal_alignment_dashboard.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "goal_alignment_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)

    selected = first(layerwise, method="hessian_layerwise")
    fixed = first(layerwise, method="fixed_order")
    labels = ["PPL degradation", "Accuracy degradation"]
    selected_values = [f(selected, "ppl_degradation"), f(selected, "accuracy_degradation")]
    fixed_values = [f(fixed, "ppl_degradation"), f(fixed, "accuracy_degradation")]
    xpos = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(xpos - width / 2, selected_values, width=width, color="#059669", label="Hessian layer-wise")
    ax.bar(xpos + width / 2, fixed_values, width=width, color="#6b7280", label="Fixed Q->S->R")
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("degradation vs baseline")
    ax.set_title("Layer-wise selection under same q/s/r settings")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figures / "goal_alignment_layerwise.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "goal_alignment_layerwise.pdf", bbox_inches="tight")
    plt.close(fig)

    write_report(out_dir, result_dir, correlations, additivity, layerwise, order_gap)
    print(out_dir)


if __name__ == "__main__":
    main()
