#!/usr/bin/env python3
"""Render an SPQ-like fixed-vs-guided summary as a dependency-free SVG."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("results/pretrained_orthogonality_summary_20260624_qwen"),
        help="Directory containing spq_execution_summary.csv.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output SVG path.")
    return parser.parse_args()


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def text(x: float, y: float, label: str, *, size: int = 13, anchor: str = "start", weight: str = "400", color: str = "#222") -> str:
    safe = html.escape(label)
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}">{safe}</text>'


def rect(x: float, y: float, width: float, height: float, color: str, *, opacity: float = 1.0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0):.1f}" height="{height:.1f}" fill="{color}" opacity="{opacity:.2f}"/>'


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#999", width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"/>'


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_svg(rows: list[dict[str, str]]) -> str:
    width, height = 1180, 760
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(48, 42, "SPQ-like fixed recipe vs Hessian-guided SPQ", size=24, weight="700"),
        text(48, 70, "Same nominal bits/keep/rank; LoRA rows use equal rank and step budgets. Lower PPL degradation is better.", size=13, color="#555"),
    ]

    colors = {
        "fixed_no": "#6b7c93",
        "guided_no": "#167f74",
        "fixed_lora": "#b45b0b",
        "guided_lora": "#7c2bd6",
    }
    labels = [
        ("fixed_spq_no_lora_ppl", "fixed no-LoRA", "fixed_no"),
        ("guided_spq_no_lora_ppl", "guided no-LoRA", "guided_no"),
        ("fixed_spq_lora_ppl", "fixed LoRA", "fixed_lora"),
        ("guided_spq_lora_ppl", "guided LoRA", "guided_lora"),
    ]

    out.append(text(48, 112, "A. Strategy PPL degradation, log10 scale", size=16, weight="700"))
    chart_x, chart_y, chart_w = 250, 132, 520
    bar_h, gap = 18, 10
    all_degs: list[float] = []
    for row in rows:
        base = to_float(row["baseline_ppl"])
        for col, _label, _key in labels:
            all_degs.append(max(to_float(row[col]) - base, 0.0))
    max_log = max(math.log10(v + 1.0) for v in all_degs if math.isfinite(v))
    max_axis_log = max(1, math.ceil(max_log))
    for tick in range(0, max_axis_log + 1):
        x = chart_x + chart_w * tick / max_axis_log
        out.append(line(x, chart_y - 18, x, chart_y + len(rows) * 136 - 12, "#e3e6ea"))
        out.append(text(x, chart_y - 24, str(int(10**tick - 1)), size=10, anchor="middle", color="#666"))
    y = chart_y
    for row in rows:
        run = row["run"].replace(" SPQ smoke", "")
        base = to_float(row["baseline_ppl"])
        out.append(text(48, y + 10, run, size=14, weight="700"))
        out.append(text(48, y + 30, f"base PPL {fmt(base)}", size=11, color="#666"))
        for col, label, key in labels:
            ppl = to_float(row[col])
            deg = max(ppl - base, 0.0)
            bar_w = chart_w * math.log10(deg + 1.0) / max_axis_log if max_axis_log > 0 else 0
            out.append(text(112, y + 14, label, size=11, color="#333"))
            out.append(rect(chart_x, y, bar_w, bar_h, colors[key]))
            out.append(text(chart_x + bar_w + 8, y + 14, f"PPL {fmt(ppl)} / deg {fmt(deg)}", size=11, color="#333"))
            y += bar_h + gap
        y += 24

    out.append(text(820, 112, "B. Guided PPL reduction vs fixed recipe", size=16, weight="700"))
    pct_x, pct_y, pct_w = 820, 136, 260
    out.append(line(pct_x, pct_y - 12, pct_x, pct_y + 138, "#999"))
    out.append(line(pct_x + pct_w, pct_y - 12, pct_x + pct_w, pct_y + 138, "#ddd"))
    out.append(text(pct_x, pct_y - 20, "0%", size=10, anchor="middle", color="#666"))
    out.append(text(pct_x + pct_w, pct_y - 20, "100%", size=10, anchor="middle", color="#666"))
    y = pct_y
    for row in rows:
        run = row["run"].replace(" SPQ smoke", "")
        out.append(text(820, y - 6, run, size=13, weight="700"))
        pairs = [
            ("no-LoRA", to_float(row["fixed_spq_no_lora_ppl"]), to_float(row["guided_spq_no_lora_ppl"]), "guided_no"),
            ("LoRA", to_float(row["fixed_spq_lora_ppl"]), to_float(row["guided_spq_lora_ppl"]), "guided_lora"),
        ]
        for label, fixed, guided, key in pairs:
            reduction = max((fixed - guided) / fixed, 0.0) if fixed > 0 else 0.0
            out.append(text(820, y + 18, label, size=11, color="#333"))
            out.append(rect(880, y + 4, pct_w * reduction, 18, colors[key]))
            out.append(text(884 + pct_w * reduction, y + 18, f"{reduction * 100:.1f}%", size=11, color="#333"))
            y += 30
        y += 18

    out.append(text(820, 360, "C. Diagnostic correlations", size=16, weight="700"))
    corr_labels = [
        ("rho_additivity", "rho_H vs add."),
        ("rho_ppl", "rho_H vs PPL"),
        ("taylor_vs_loss", "Taylor vs loss"),
        ("frobenius_vs_loss", "Frob. vs loss"),
        ("spectrum_order_entropy", "Spec. vs order"),
        ("order_disagreement", "Disagree vs order"),
    ]
    corr_x, corr_y, corr_w = 930, 392, 190
    out.append(line(corr_x, corr_y - 18, corr_x, corr_y + len(corr_labels) * 46 + 10, "#999"))
    out.append(line(corr_x + corr_w, corr_y - 18, corr_x + corr_w, corr_y + len(corr_labels) * 46 + 10, "#ddd"))
    out.append(text(corr_x, corr_y - 26, "0", size=10, anchor="middle", color="#666"))
    out.append(text(corr_x + corr_w, corr_y - 26, "1", size=10, anchor="middle", color="#666"))
    y = corr_y
    for col, label in corr_labels:
        out.append(text(820, y + 13, label, size=11, color="#333"))
        offset = 0
        for row in rows:
            val = max(to_float(row[col]), 0.0)
            color = "#167f74" if "Qwen" in row["run"] else "#6b7c93"
            out.append(rect(corr_x, y + offset, corr_w * min(val, 1.0), 12, color))
            out.append(text(corr_x + corr_w * min(val, 1.0) + 6, y + offset + 10, fmt(val, 2), size=10, color="#333"))
            offset += 15
        y += 46
    out.append(rect(820, 696, 14, 10, "#6b7c93"))
    out.append(text(840, 705, "Pythia-70M", size=11, color="#444"))
    out.append(rect(930, 696, 14, 10, "#167f74"))
    out.append(text(950, 705, "Qwen2.5-1.5B", size=11, color="#444"))

    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    csv_path = args.summary_dir / "spq_execution_summary.csv"
    output = args.output or args.summary_dir / "figures" / "spq_fixed_vs_guided_summary.svg"
    rows = load_rows(csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_svg(rows), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
