from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DPI = 300
FONT_SIZE = 10


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1,
            "legend.frameon": False,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
        }
    )


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", format="pdf", dpi=DPI)
    fig.savefig(out_dir / f"{name}.png", format="png", dpi=DPI)
    plt.close(fig)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt(value: object, digits: int = 4) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value_f):
        return "n/a"
    return f"{value_f:.{digits}g}"


def plot_spectra(result_dir: Path, out_dir: Path) -> None:
    df = read_csv(result_dir / "phase1" / "metrics" / "spectra.csv")
    if df.empty:
        return
    setup_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for module_type, group in df.groupby("module_type"):
        curve = group.groupby("rank_index")["singular_value"].median().reset_index()
        ax.loglog(curve["rank_index"], curve["singular_value"], linewidth=1.6, label=module_type)
    ax.set_xlabel("Rank index")
    ax.set_ylabel("Median singular value")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2)
    save(fig, out_dir, "spectral_decay_by_module")


def plot_approximation_errors(result_dir: Path, out_dir: Path) -> None:
    df = read_csv(result_dir / "phase1" / "metrics" / "approximation_errors.csv")
    if df.empty:
        return
    setup_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    grouped = df.groupby(["method", "compression_ratio_target"])["relative_weight_error"].median().reset_index()
    for method, group in grouped.groupby("method"):
        ax.plot(group["compression_ratio_target"], group["relative_weight_error"], marker="o", linewidth=1.6, label=method)
    ax.set_xlabel("Target compression ratio")
    ax.set_ylabel("Median relative weight error")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, out_dir, "structured_approximation_error")


def plot_residuals(result_dir: Path, out_dir: Path) -> None:
    df = read_csv(result_dir / "phase1" / "metrics" / "residual_metrics.csv")
    if df.empty:
        return
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    grouped = df.groupby(["residual_type", "residual_fraction"])["relative_weight_error_after_residual"].median().reset_index()
    for residual_type, group in grouped.groupby("residual_type"):
        axes[0].plot(group["residual_fraction"], group["relative_weight_error_after_residual"], marker="o", linewidth=1.4, label=residual_type)
    axes[0].set_xlabel("Residual fraction")
    axes[0].set_ylabel("Median weight error")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    if "residual_top_1pct_l1_frac" in df:
        summary = df.groupby("module_type")["residual_top_1pct_l1_frac"].median().sort_values()
        axes[1].barh(summary.index.astype(str), summary.values)
        axes[1].set_xlabel("Top 1% residual L1 fraction")
    fig.tight_layout()
    save(fig, out_dir, "residual_budget_and_concentration")


def plot_phase2(result_dir: Path, out_dir: Path) -> None:
    df = read_csv(result_dir / "phase2" / "metrics" / "activation_reconstruction.csv")
    if df.empty:
        return
    setup_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for method, group in df.groupby("method"):
        ax.scatter(group["relative_weight_error_after_residual"], group["relative_activation_error_after_residual"], s=18, alpha=0.65, label=method)
    ax.set_xlabel("Relative weight error")
    ax.set_ylabel("Relative activation error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, out_dir, "phase2_weight_vs_activation_error")


def plot_phase3(result_dir: Path, out_dir: Path) -> None:
    df = read_csv(result_dir / "phase3" / "metrics" / "compression_performance.csv")
    if df.empty:
        return
    df = df[df["stage"] != "baseline"]
    if df.empty:
        return
    setup_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for stage, group in df.groupby("stage"):
        curve = group.groupby("effective_replaced_compression")["perplexity"].median().reset_index().sort_values("effective_replaced_compression")
        ax.plot(curve["effective_replaced_compression"], curve["perplexity"], marker="o", linewidth=1.6, label=stage)
    ax.set_xlabel("Effective compression on replaced weights")
    ax.set_ylabel("Perplexity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, out_dir, "phase3_compression_vs_perplexity")


def plot_phase4(result_dir: Path, out_dir: Path) -> None:
    df = read_csv(result_dir / "phase4" / "metrics" / "peft_performance.csv")
    if df.empty:
        return
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    x_col = "adapter_params" if "adapter_params" in df else "budget"
    grouped = df.groupby(["method", x_col])["perplexity"].median().reset_index()
    for method, group in grouped.groupby("method"):
        axes[0].plot(group[x_col], group["perplexity"], marker="o", linewidth=1.4, label=method)
    axes[0].set_xlabel("Actual adapter parameters")
    axes[0].set_ylabel("Perplexity")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    rank_df = df[df["rank"].astype(str) != ""].copy()
    if not rank_df.empty:
        rank_df["rank"] = rank_df["rank"].astype(float)
        for method, group in rank_df.groupby("method"):
            curve = group.groupby("rank")["perplexity"].median().reset_index()
            axes[1].plot(curve["rank"], curve["perplexity"], marker="o", linewidth=1.4, label=method)
        axes[1].set_xlabel("Adapter rank")
        axes[1].set_ylabel("Perplexity")
        axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, "phase4_peft_budget_and_rank")


def plot_phase5(result_dir: Path, out_dir: Path) -> None:
    qdf = read_csv(result_dir / "phase5" / "metrics" / "structured_quantization.csv")
    rdf = read_csv(result_dir / "phase5" / "metrics" / "rotation_outliers.csv")
    if qdf.empty and rdf.empty:
        return
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    if not qdf.empty:
        grouped = qdf.groupby(["rotation_type", "bit_width"])["relative_structured_quantized_error"].median().reset_index()
        for rotation, group in grouped.groupby("rotation_type"):
            axes[0].plot(group["bit_width"], group["relative_structured_quantized_error"], marker="o", linewidth=1.4, label=rotation)
        axes[0].invert_xaxis()
        axes[0].set_xlabel("Bit width")
        axes[0].set_ylabel("Structured quantized error")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=7)
    if not rdf.empty:
        cols = ["in_channel_max_over_median_before", "in_channel_max_over_median_after"]
        summary = rdf.groupby("rotation_type")[cols].median()
        x = np.arange(len(summary.index))
        width = 0.35
        axes[1].bar(x - width / 2, summary[cols[0]], width, label="before")
        axes[1].bar(x + width / 2, summary[cols[1]], width, label="after")
        axes[1].set_xticks(x, labels=summary.index, rotation=25, ha="right")
        axes[1].set_ylabel("Input-channel max/median")
        axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "phase5_rotation_quantization")


def plot_structure_heatmap(result_dir: Path, out_dir: Path) -> None:
    df = read_csv(result_dir / "phase1" / "metrics" / "approximation_errors.csv")
    if df.empty:
        return
    setup_style()
    pivot = df.groupby(["module_type", "method"])["relative_weight_error"].median().unstack()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * len(pivot.columns)), max(3.0, 0.38 * len(pivot.index))))
    values = pivot.values
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("Structure")
    ax.set_ylabel("Module type")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Median weight error")
    fig.tight_layout()
    save(fig, out_dir, "layer_type_structure_heatmap")


def write_report(result_dir: Path) -> None:
    lines = ["# Structured Qwen2.5-1.5B Compression Report", ""]

    phase3 = read_csv(result_dir / "phase3" / "metrics" / "compression_performance.csv")
    zero_shot = read_csv(result_dir / "phase3" / "metrics" / "zero_shot.csv")
    if not phase3.empty:
        baseline = phase3[phase3["stage"] == "baseline"]
        if not baseline.empty:
            lines.append(f"- Baseline perplexity: {_fmt(baseline.iloc[0].get('perplexity'))}.")
        compressed = phase3[phase3["stage"] != "baseline"]
        if not compressed.empty:
            best = compressed.loc[compressed["perplexity"].astype(float).idxmin()]
            lines.append(
                "- Best compressed PPL row: "
                f"stage={best.get('stage')}, ratio={best.get('compression_ratio_target')}, "
                f"residual={best.get('residual_fraction')}, ppl={_fmt(best.get('perplexity'))}."
            )
            stages = ", ".join(sorted(str(x) for x in compressed["stage"].dropna().unique()))
            lines.append(f"- Completed replacement stages: {stages}.")
    if not zero_shot.empty and "status" in zero_shot:
        ok_rows = zero_shot[zero_shot["status"] == "ok"]
        if not ok_rows.empty and "accuracy" in ok_rows:
            lines.append(f"- Mean reported zero-shot accuracy: {_fmt(ok_rows['accuracy'].astype(float).mean())}.")

    phase1 = read_csv(result_dir / "phase1" / "metrics" / "approximation_errors.csv")
    if not phase1.empty:
        lines.append("")
        lines.append("## Phase 1")
        best_rows = []
        for module_type, group in phase1.groupby("module_type"):
            idx = group["relative_weight_error"].astype(float).idxmin()
            best = group.loc[idx]
            best_rows.append(f"{module_type}: {best.get('method')}@{best.get('compression_ratio_target')}x")
        if best_rows:
            lines.append("- Best weight-error structure by module type: " + "; ".join(best_rows) + ".")

    phase2 = read_csv(result_dir / "phase2" / "metrics" / "activation_reconstruction.csv")
    if not phase2.empty:
        lines.append("")
        lines.append("## Phase 2")
        metric = "relative_activation_error_after_residual"
        if metric in phase2:
            best = phase2.loc[phase2[metric].astype(float).idxmin()]
            lines.append(
                "- Best activation reconstruction row: "
                f"module={best.get('module_type')}, method={best.get('method')}, "
                f"residual={best.get('residual_type')}, activation_error={_fmt(best.get(metric))}."
            )

    phase4 = read_csv(result_dir / "phase4" / "metrics" / "peft_performance.csv")
    if not phase4.empty:
        lines.append("")
        lines.append("## Phase 4")
        methods = ", ".join(sorted(str(x) for x in phase4["method"].dropna().unique()))
        best = phase4.loc[phase4["perplexity"].astype(float).idxmin()]
        lines.append(f"- Adapter methods covered: {methods}.")
        lines.append(
            "- Best adapter row: "
            f"method={best.get('method')}, rank={_fmt(best.get('rank'), digits=3)}, "
            f"params={best.get('adapter_params')}, budget_per_module={best.get('budget_per_module', best.get('budget'))}, "
            f"ppl={_fmt(best.get('perplexity'))}."
        )

    phase5_rot = read_csv(result_dir / "phase5" / "metrics" / "rotation_outliers.csv")
    phase5_quant = read_csv(result_dir / "phase5" / "metrics" / "quantization_errors.csv")
    if not phase5_rot.empty or not phase5_quant.empty:
        lines.append("")
        lines.append("## Phase 5")
    if not phase5_rot.empty:
        cols = ["rotation_type", "in_channel_max_over_median_before", "in_channel_max_over_median_after"]
        if all(col in phase5_rot for col in cols):
            row = phase5_rot.loc[phase5_rot["in_channel_max_over_median_after"].astype(float).idxmin()]
            lines.append(
                "- Best input-channel outlier reduction row: "
                f"rotation={row.get('rotation_type')}, "
                f"before={_fmt(row.get('in_channel_max_over_median_before'))}, "
                f"after={_fmt(row.get('in_channel_max_over_median_after'))}, "
                f"norm_change={_fmt(row.get('relative_norm_change', 0.0))}."
            )
    if not phase5_quant.empty:
        row = phase5_quant.loc[phase5_quant["relative_quantization_error"].astype(float).idxmin()]
        lines.append(
            "- Lowest direct quantization-error row: "
            f"rotation={row.get('rotation_type')}, bits={row.get('bit_width')}, "
            f"error={_fmt(row.get('relative_quantization_error'))}."
        )

    lines.append("")
    lines.append(
        "Detailed CSV artifacts are under `phase1/metrics`, `phase2/metrics`, "
        "`phase3/metrics`, `phase4/metrics`, and `phase5/metrics`; figures are under `figures/`."
    )
    (result_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot structured compression experiment outputs.")
    parser.add_argument("--result-dir", default="results/structured_qwen25_1p5b")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    out_dir = result_dir / "figures"
    plot_spectra(result_dir, out_dir)
    plot_approximation_errors(result_dir, out_dir)
    plot_residuals(result_dir, out_dir)
    plot_phase2(result_dir, out_dir)
    plot_phase3(result_dir, out_dir)
    plot_phase4(result_dir, out_dir)
    plot_phase5(result_dir, out_dir)
    plot_structure_heatmap(result_dir, out_dir)
    write_report(result_dir)


if __name__ == "__main__":
    main()
