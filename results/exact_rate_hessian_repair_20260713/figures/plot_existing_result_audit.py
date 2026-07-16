from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Q", "Q+L", "Q+S", "Q+S+L"]
RUNS = ["Pythia-70M", "Pythia-160M", "Qwen2-7B attention-only", "Qwen2-7B attn+MLP"]
COLORS = {
    "nominal": "#4C78A8",
    "entropy": "#F58518",
    "csr": "#E45756",
    "attention": "#54A24B",
    "attn+MLP": "#B279A2",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    setup_style()
    payload = read_csv(ROOT / "payload_audit.csv")
    stability = read_csv(ROOT / "strategy_stability.csv")
    interactions = read_csv(ROOT / "hessian_interactions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))

    ax = axes[0, 0]
    x = np.arange(len(METHODS) * 2, dtype=float)
    labels: list[str] = []
    nominal: list[float] = []
    entropy: list[float] = []
    csr: list[float] = []
    for run_short, run in (("Attn", "Qwen2-7B attention"), ("Attn+MLP", "Qwen2-7B attn+MLP")):
        by_method = {row["method"]: row for row in payload if row["run"] == run}
        for method in METHODS:
            row = by_method[method]
            labels.append(f"{run_short}\n{method}")
            nominal.append(float(row["nominal_ratio"]))
            entropy.append(float(row["entropy_lower_bound_ratio"]))
            csr.append(float(row["csr16_ratio"]))
    width = 0.24
    ax.bar(x - width, nominal, width=width, color=COLORS["nominal"], label="Nominal")
    ax.bar(x, entropy, width=width, color=COLORS["entropy"], label="Entropy lower bound")
    ax.bar(x + width, csr, width=width, color=COLORS["csr"], label="CSR16")
    ax.axhline(0.258, color="black", linewidth=1.0, linestyle="--", label="Target 0.258")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Payload / dense FP16")
    ax.set_title("(a) Metadata changes the matched-rate claim", loc="left")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0.245, 0.270)

    ax = axes[0, 1]
    matrix = np.full((len(RUNS), len(METHODS)), np.nan)
    for row in stability:
        if row["run"] in RUNS and row["method"] in METHODS:
            matrix[RUNS.index(row["run"]), METHODS.index(row["method"])] = float(row["ppl_delta"])
    bound = max(float(np.nanmax(np.abs(matrix))), 1.0)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS)
    ax.set_yticks(range(len(RUNS)))
    ax.set_yticklabels(RUNS)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center", color="white" if abs(value) > 0.45 * bound else "black", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("PPL delta")
    ax.set_title("(b) Q+S+L wins only 2/4 committed runs", loc="left")

    ax = axes[1, 0]
    pairs = ["Qerr,Sres", "Qerr,Lres", "Q+S err,Lres", "Sres,Lres"]
    markers = {"Qwen2-7B attention": "o", "Qwen2-7B attn+MLP": "s"}
    colors = {"Q+S": COLORS["attention"], "Q+S+L": COLORS["attn+MLP"]}
    for run in ("Qwen2-7B attention", "Qwen2-7B attn+MLP"):
        for method in ("Q+S", "Q+S+L"):
            subset = [row for row in interactions if row["run"] == run and row["method"] == method]
            if not subset:
                continue
            by_pair = {row["pair"]: float(row["rho_h"]) for row in subset}
            used = [pair for pair in pairs if pair in by_pair]
            ax.scatter(
                [pairs.index(pair) for pair in used],
                [by_pair[pair] for pair in used],
                marker=markers[run],
                color=colors[method],
                s=48,
                label=f"{run.replace('Qwen2-7B ', '')}, {method}",
                edgecolor="white",
                linewidth=0.5,
            )
    ax.axhspan(-0.05, 0.05, color="#72B7B2", alpha=0.18, label="|rho_H| <= 0.05")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=20, ha="right")
    ax.set_ylabel("Hessian cosine rho_H")
    ax.set_title("(c) Negative rho is repair cancellation, not orthogonality", loc="left")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    for run, color, marker in (
        ("Qwen2-7B attention", COLORS["attention"], "o"),
        ("Qwen2-7B attn+MLP", COLORS["attn+MLP"], "s"),
    ):
        subset = [row for row in payload if row["run"] == run]
        for row in subset:
            x_value = float(row["entropy_lower_bound_ratio"])
            y_value = float(row["ppl_delta"])
            ax.scatter(x_value, y_value, color=color, marker=marker, s=50, edgecolor="white", linewidth=0.5)
            ax.annotate(row["method"], (x_value, y_value), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.axvline(0.258, color="black", linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Entropy-lower-bound payload / dense FP16")
    ax.set_ylabel("PPL delta")
    ax.set_title("(d) Endpoint quality must be read with actual payload", loc="left")

    fig.tight_layout()
    output_pdf = Path(__file__).with_name("existing_result_audit.pdf")
    output_png = Path(__file__).with_name("existing_result_audit.png")
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(output_pdf)
    print(output_png)


if __name__ == "__main__":
    main()
