#!/usr/bin/env python3
"""Summarize rotation-quantization and low-loss triple-stack smoke results."""

from __future__ import annotations

import argparse
import csv
import html
import statistics
from pathlib import Path


RUNS = [
    ("Pythia-70M", "pretrained_orthogonality_pythia70m_rotation_low_loss_smoke_20260625"),
    ("Qwen2.5-1.5B", "pretrained_orthogonality_qwen25_1p5b_rotation_low_loss_smoke_20260625"),
]
ZERO_SHOT_CHECK = "pretrained_orthogonality_pythia70m_low_loss_zeroshot100_20260625"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def median(rows: list[dict[str, str]], method: str, key: str) -> float:
    values = [f(row, key) for row in rows if row.get("q_method") == method]
    return statistics.median(values) if values else float("nan")


def collect(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, directory in RUNS:
        run_dir = root / "results" / directory
        strategy = {row["strategy"]: row for row in read_csv(run_dir / "metrics" / "strategy_performance.csv")}
        rotation = read_csv(run_dir / "metrics" / "rotation_quantization.csv")
        base = strategy["baseline"]
        default = strategy["fixed_qsr_default"]
        rotated = strategy["fixed_qsr_rotated_q"]
        low = strategy["low_loss_triple_stack"]
        rows.append(
            {
                "run": label,
                "run_dir": directory,
                "baseline_ppl": f(base, "perplexity"),
                "fixed_qsr_default_ppl": f(default, "perplexity"),
                "fixed_qsr_rotated_q_ppl": f(rotated, "perplexity"),
                "rotated_minus_default_ppl": f(rotated, "perplexity") - f(default, "perplexity"),
                "low_loss_ppl": f(low, "perplexity"),
                "low_loss_ppl_degradation": f(low, "ppl_degradation"),
                "low_loss_benchmark_drop_percent": f(low, "benchmark_drop_percent"),
                "low_loss_pass": low.get("lossless_pass", ""),
                "low_loss_order": low.get("order", ""),
                "low_loss_q_method": low.get("q_method", ""),
                "low_loss_s_method": low.get("s_method", ""),
                "low_loss_r_method": low.get("r_method", ""),
                "median_rtn_weight_error": median(rotation, "rtn", "relative_weight_error"),
                "median_rotated_weight_error": median(rotation, "rotated_rtn", "relative_weight_error"),
                "median_rtn_hessian_cost": median(rotation, "rtn", "hessian_self_cost"),
                "median_rotated_hessian_cost": median(rotation, "rotated_rtn", "hessian_self_cost"),
                "median_rtn_outlier_ratio": median(rotation, "rtn", "input_channel_max_over_median_quant_basis"),
                "median_rotated_outlier_ratio": median(rotation, "rotated_rtn", "input_channel_max_over_median_quant_basis"),
            }
        )
    structured_rot = read_csv(root / "results" / "structured_qwen25_1p5b_goal_smoke_20260606_024653" / "phase5" / "metrics" / "rotation_outliers.csv")
    structured_quant = read_csv(root / "results" / "structured_qwen25_1p5b_goal_smoke_20260606_024653" / "phase5" / "metrics" / "quantization_errors.csv")
    hadamard_rot = next((row for row in structured_rot if row.get("rotation_type") == "hadamard"), {})
    none_quant = next((row for row in structured_quant if row.get("rotation_type") == "none" and row.get("bit_width") == "4"), {})
    hadamard_quant = next((row for row in structured_quant if row.get("rotation_type") == "hadamard" and row.get("bit_width") == "4"), {})
    rows.append(
        {
            "run": "Structured Qwen phase5 down_proj",
            "run_dir": "structured_qwen25_1p5b_goal_smoke_20260606_024653",
            "structured_outlier_before": f(hadamard_rot, "in_channel_max_over_median_before"),
            "structured_outlier_after": f(hadamard_rot, "in_channel_max_over_median_after"),
            "structured_outlier_count_before": f(hadamard_rot, "in_channel_outlier_count_before"),
            "structured_outlier_count_after": f(hadamard_rot, "in_channel_outlier_count_after"),
            "structured_4bit_error_none": f(none_quant, "relative_quantization_error"),
            "structured_4bit_error_hadamard": f(hadamard_quant, "relative_quantization_error"),
        }
    )
    zs_dir = root / "results" / ZERO_SHOT_CHECK
    if zs_dir.exists():
        strategy = {row["strategy"]: row for row in read_csv(zs_dir / "metrics" / "strategy_performance.csv")}
        baseline = strategy.get("baseline", {})
        low = strategy.get("low_loss_triple_stack", {})
        rows.append(
            {
                "run": "Pythia-70M ARC-Easy100 low-loss check",
                "run_dir": ZERO_SHOT_CHECK,
                "benchmark_task": "arc_easy",
                "benchmark_examples": 100,
                "baseline_zero_shot_accuracy": f(baseline, "zero_shot_accuracy"),
                "low_loss_zero_shot_accuracy": f(low, "zero_shot_accuracy"),
                "low_loss_zero_shot_degradation": f(low, "zero_shot_accuracy_degradation"),
                "low_loss_benchmark_drop_percent": f(low, "benchmark_drop_percent"),
                "low_loss_pass": low.get("lossless_pass", ""),
                "low_loss_order": low.get("order", ""),
                "low_loss_q_method": low.get("q_method", ""),
                "low_loss_s_method": low.get("s_method", ""),
                "low_loss_r_method": low.get("r_method", ""),
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def t(x: float, y: float, label: str, *, size: int = 12, weight: str = "400", anchor: str = "start", color: str = "#222") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{html.escape(label)}</text>'


def r(x: float, y: float, w: float, h: float, color: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" fill="{color}"/>'


def render_svg(path: Path, rows: list[dict[str, object]]) -> None:
    smoke = rows[:2]
    width, height = 1100, 620
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        t(44, 42, "Rotation quantization and low-loss Q+S+R smoke summary", size=22, weight="700"),
        t(44, 68, "Lower is better for all bars. Lossless threshold: benchmark drop < 1%.", size=13, color="#555"),
    ]
    max_err = max(float(row["median_rtn_weight_error"]) for row in smoke)
    max_h = max(float(row["median_rtn_hessian_cost"]) for row in smoke)
    max_out = max(float(row["median_rtn_outlier_ratio"]) for row in smoke)
    panels = [
        ("median relative Q error", "median_rtn_weight_error", "median_rotated_weight_error", max_err, 110),
        ("median Hessian Q cost", "median_rtn_hessian_cost", "median_rotated_hessian_cost", max_h, 260),
        ("median input outlier ratio", "median_rtn_outlier_ratio", "median_rotated_outlier_ratio", max_out, 410),
    ]
    for title, rtn_key, rot_key, max_value, y0 in panels:
        out.append(t(44, y0 - 14, title, size=15, weight="700"))
        for idx, row in enumerate(smoke):
            y = y0 + idx * 48
            out.append(t(44, y + 13, str(row["run"]), size=12))
            rtn_w = 270 * float(row[rtn_key]) / max(max_value, 1e-12)
            rot_w = 270 * float(row[rot_key]) / max(max_value, 1e-12)
            out.append(r(210, y, rtn_w, 14, "#6b7c93"))
            out.append(r(210, y + 18, rot_w, 14, "#167f74"))
            out.append(t(488, y + 12, f"RTN {float(row[rtn_key]):.4g}", size=10, color="#444"))
            out.append(t(488, y + 30, f"Rot {float(row[rot_key]):.4g}", size=10, color="#444"))
    out.append(t(650, 96, "Low-loss triple-stack benchmark drop", size=15, weight="700"))
    for idx, row in enumerate(smoke):
        y = 126 + idx * 76
        out.append(t(650, y, str(row["run"]), size=13, weight="700"))
        drop = float(row["low_loss_benchmark_drop_percent"])
        out.append(r(650, y + 16, 300 * min(drop / 1.0, 1.0), 18, "#7c2bd6"))
        out.append(r(650 + 300, y + 16, 2, 18, "#111"))
        out.append(t(960, y + 30, f"{drop:.4f}% pass={row['low_loss_pass']}", size=11))
        out.append(t(650, y + 52, f"order={row['low_loss_order']} q={row['low_loss_q_method']} PPL deg={float(row['low_loss_ppl_degradation']):.4f}", size=11, color="#555"))
    structured = rows[2]
    out.append(t(650, 320, "Structured Qwen phase5 Hadamard check", size=15, weight="700"))
    out.append(t(650, 352, f"input outlier max/median: {float(structured['structured_outlier_before']):.3f} -> {float(structured['structured_outlier_after']):.3f}", size=12))
    out.append(t(650, 378, f"outlier count: {int(float(structured['structured_outlier_count_before']))} -> {int(float(structured['structured_outlier_count_after']))}", size=12))
    out.append(t(650, 404, f"4-bit quantization error: {float(structured['structured_4bit_error_none']):.4f} -> {float(structured['structured_4bit_error_hadamard']):.4f}", size=12))
    out.append(r(650, 432, 300 * float(structured["structured_4bit_error_none"]) / 0.2, 18, "#6b7c93"))
    out.append(r(650, 456, 300 * float(structured["structured_4bit_error_hadamard"]) / 0.2, 18, "#167f74"))
    out.append(t(960, 446, "none", size=11))
    out.append(t(960, 470, "Hadamard", size=11))
    if len(rows) > 3:
        zs = rows[3]
        out.append(t(650, 520, "Zero-shot benchmark check", size=15, weight="700"))
        out.append(t(650, 548, f"ARC-Easy {int(float(zs['benchmark_examples']))} examples: baseline {float(zs['baseline_zero_shot_accuracy']):.3f} -> low-loss {float(zs['low_loss_zero_shot_accuracy']):.3f}", size=12))
        out.append(t(650, 574, f"benchmark drop {float(zs['low_loss_benchmark_drop_percent']):.4f}% pass={zs['low_loss_pass']} order={zs['low_loss_order']} q={zs['low_loss_q_method']}", size=12))
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--summary-dir", type=Path, default=Path("results/pretrained_orthogonality_summary_20260624_qwen"))
    args = parser.parse_args()
    rows = collect(args.root)
    write_summary_csv(args.root / args.summary_dir / "rotation_low_loss_summary.csv", rows)
    render_svg(args.root / args.summary_dir / "figures" / "rotation_low_loss_summary.svg", rows)
    print(args.root / args.summary_dir)


if __name__ == "__main__":
    main()
