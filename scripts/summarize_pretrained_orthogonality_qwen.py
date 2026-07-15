from __future__ import annotations

import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "pretrained_orthogonality_summary_20260624_qwen"
FIGDIR = OUT / "figures"


RUNS: list[dict[str, Any]] = [
    {
        "run": "70M_default_4bit_6mods",
        "dir": "pretrained_orthogonality_pythia70m_20260623_v2",
        "params": 70_426_624,
        "param_label": "70.4M",
        "model_family": "Pythia",
        "purpose": "mild compression sanity check with fewer modules; tests whether the framework is useful before aggressive degradation",
        "server_source": "236 GPU server, Hugging Face model id",
        "comparability": "Pythia-family; same text source and 8 zero-shot examples/task as other Pythia runs",
    },
    {
        "run": "70M_mid_3bit_12mods",
        "dir": "pretrained_orthogonality_pythia70m_mid_20260623_v2",
        "params": 70_426_624,
        "param_label": "70.4M",
        "model_family": "Pythia",
        "purpose": "main 70M configuration; 12-module coverage for stronger correlation and strategy comparisons",
        "server_source": "236 GPU server, Hugging Face model id",
        "comparability": "Matched mid compression setting for Pythia 70M/160M; 8 zero-shot examples/task",
    },
    {
        "run": "70M_strong_2bit_12mods",
        "dir": "pretrained_orthogonality_pythia70m_strong_20260623_v2",
        "params": 70_426_624,
        "param_label": "70.4M",
        "model_family": "Pythia",
        "purpose": "stress test; checks whether overlap/additivity signals become clearer under larger perturbations",
        "server_source": "236 GPU server, Hugging Face model id",
        "comparability": "Same model as 70M mid but stronger 2-bit/0.4 keep/rank compression",
    },
    {
        "run": "160M_mid_3bit_12mods",
        "dir": "pretrained_orthogonality_pythia160m_mid_20260624",
        "params": 162_322_944,
        "param_label": "162.3M",
        "model_family": "Pythia",
        "purpose": "larger-parameter validation with the same mid configuration as 70M_mid",
        "server_source": "236 GPU server, Hugging Face model id",
        "comparability": "Matched Pythia mid compression setting; 8 zero-shot examples/task",
    },
    {
        "run": "Qwen2.5_1.5B_mid_3bit_12mods",
        "dir": "pretrained_orthogonality_qwen25_1p5b_mid_20260624",
        "params": "",
        "param_label": "~1.5B",
        "model_family": "Qwen2.5",
        "purpose": "larger model-family validation on 210; tests whether conclusions survive outside Pythia",
        "server_source": "210 A800 server, local path /home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B",
        "comparability": "Qualitative cross-family validation; shorter evaluation budget: 4 zero-shot examples/task and 508 PPL tokens; parameter count is an approximate model-family label because this run artifact predates measured parameter-count logging",
    },
]


CORR_FAMILIES = [
    ("additivity", "rho additivity"),
    ("real_ppl", "rho PPL"),
    ("real_zero_shot", "rho zero-shot"),
    ("taylor", "Taylor vs loss"),
    ("frobenius_baseline", "Frobenius vs loss"),
    ("parameter_cosine_baseline", "Param cosine vs add."),
    ("activation_reconstruction_baseline", "Act recon vs loss"),
    ("trace_only_baseline", "Trace-only vs loss"),
    ("spectrum_order_entropy", "Spectrum entropy vs order"),
    ("order_gap", "H-overlap vs order"),
]


COLORS = {
    "70M_default_4bit_6mods": "#4C78A8",
    "70M_mid_3bit_12mods": "#59A14F",
    "70M_strong_2bit_12mods": "#E15759",
    "160M_mid_3bit_12mods": "#F28E2B",
    "Qwen2.5_1.5B_mid_3bit_12mods": "#B07AA1",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def flt(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    ax = abs(x)
    if ax >= 1e5 or (0 < ax < 1e-3):
        return f"{x:.3e}"
    if ax >= 100:
        return f"{x:.1f}"
    return f"{x:.4f}"


def md_table(rows: list[dict[str, Any]], cols: list[str], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(fmt(value))
            elif isinstance(value, int):
                vals.append(f"{value:,}")
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def collect_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    method_status_rows: list[dict[str, Any]] = []

    for spec in RUNS:
        metrics = RESULTS / spec["dir"] / "metrics"
        cfg = json.loads((metrics / "run_config.json").read_text(encoding="utf-8"))
        corr_map = {row["family"]: flt(row["spearman_rho"]) for row in read_csv(metrics / "correlations.csv")}
        strat = read_csv(metrics / "strategy_performance.csv")
        zbase = read_csv(metrics / "zero_shot_baseline.csv") if (metrics / "zero_shot_baseline.csv").exists() else []
        text_meta = read_csv(metrics / "text_source_metadata.csv") if (metrics / "text_source_metadata.csv").exists() else []
        add = read_csv(metrics / "additivity.csv")
        order = read_csv(metrics / "order_gap.csv")
        hcos = read_csv(metrics / "hessian_cosine.csv")
        methods = read_csv(metrics / "method_status.csv") if (metrics / "method_status.csv").exists() else []

        comp = cfg.get("compression", {})
        selected_layers = cfg.get("selected_layers", [])
        by_strategy = {row["strategy"]: row for row in strat}
        baseline = by_strategy["baseline"]
        fixed_naive = by_strategy.get("fixed_qsr_naive", {})
        fixed_default = by_strategy.get("fixed_qsr_default", {})
        slim = by_strategy.get("slim_like_srq_proxy", {})
        hessian = by_strategy.get("hessian_layerwise", {})

        for row in strat:
            out_row = dict(row)
            out_row.update(
                {
                    "run": spec["run"],
                    "model": cfg.get("model", ""),
                    "params": spec["params"],
                    "model_family": spec["model_family"],
                }
            )
            strategy_rows.append(out_row)

        for row in methods:
            out_row = dict(row)
            out_row["run"] = spec["run"]
            method_status_rows.append(out_row)

        if zbase:
            tasks = ",".join(row["task"] for row in zbase)
            examples = "; ".join(f"{row['task']}:{int(flt(row['examples']))}" for row in zbase)
            min_examples = int(min(flt(row["examples"]) for row in zbase))
        else:
            tasks = ""
            examples = ""
            min_examples = ""

        if text_meta:
            detail = "; ".join(
                f"{row['task']}:{int(flt(row['rows_used_for_text_source']))} rows fp={row['fingerprint']}"
                for row in text_meta
            )
        else:
            detail = ""

        config_rows.append(
            {
                "run": spec["run"],
                "model": cfg.get("model", ""),
                "model_family": spec["model_family"],
                "params": spec["params"],
                "param_label": spec["param_label"],
                "server_source": spec["server_source"],
                "purpose": spec["purpose"],
                "comparability_note": spec["comparability"],
                "selected_layers": len(selected_layers),
                "selected_layer_names": ", ".join(selected_layers),
                "bits": comp.get("bits", ""),
                "keep_fraction": comp.get("keep_fraction", ""),
                "rank_fraction": comp.get("rank_fraction", ""),
                "q_method": comp.get("q_method", ""),
                "s_method": comp.get("s_method", ""),
                "r_method": comp.get("r_method", ""),
                "eval_tokens": int(flt(baseline.get("tokens", 0))),
                "zero_shot_tasks": tasks,
                "zero_shot_examples_per_task": examples,
                "zero_shot_min_examples_per_task": min_examples,
                "text_source_requested": cfg.get("text_source_requested", ""),
                "text_source_used": cfg.get("text_source_used", ""),
                "text_source_detail": detail,
            }
        )

        high_rho = max(hcos, key=lambda row: flt(row["abs_rho_h"]))
        high_add = max(add, key=lambda row: flt(row["abs_additivity_error"]))
        max_gap = max(order, key=lambda row: flt(row["abs_loss_gap"]))
        summary = {
            "run": spec["run"],
            "model": cfg.get("model", ""),
            "model_family": spec["model_family"],
            "params": spec["params"],
            "param_label": spec["param_label"],
            "modules": len(selected_layers),
            "bits": comp.get("bits", ""),
            "keep_fraction": comp.get("keep_fraction", ""),
            "rank_fraction": comp.get("rank_fraction", ""),
            "eval_tokens": int(flt(baseline.get("tokens", 0))),
            "zero_shot_examples_min": min_examples,
            "baseline_ppl": flt(baseline.get("perplexity")),
            "baseline_zero_shot": flt(baseline.get("zero_shot_accuracy")),
            "hessian_layerwise_ppl": flt(hessian.get("perplexity"), float("nan")),
            "fixed_qsr_naive_ppl": flt(fixed_naive.get("perplexity"), float("nan")),
            "fixed_qsr_default_ppl": flt(fixed_default.get("perplexity"), float("nan")),
            "slim_like_srq_proxy_ppl": flt(slim.get("perplexity"), float("nan")),
            "hessian_vs_fixed_naive_ppl_delta": flt(hessian.get("perplexity"), float("nan"))
            - flt(fixed_naive.get("perplexity"), float("nan")),
            "hessian_vs_fixed_default_ppl_delta": flt(hessian.get("perplexity"), float("nan"))
            - flt(fixed_default.get("perplexity"), float("nan")),
            "hessian_vs_slim_proxy_ppl_delta": flt(hessian.get("perplexity"), float("nan"))
            - flt(slim.get("perplexity"), float("nan")),
            "highest_abs_rho_layer": high_rho["layer_short"],
            "highest_abs_rho_pair": high_rho["pair"],
            "highest_abs_rho_h": flt(high_rho["abs_rho_h"]),
            "largest_additivity_layer": high_add["layer_short"],
            "largest_additivity_pair": high_add["pair"],
            "largest_abs_additivity_error": flt(high_add["abs_additivity_error"]),
            "largest_order_gap_layer": max_gap["layer_short"],
            "largest_order_gap_orders": f"{max_gap['left_order']} vs {max_gap['right_order']}",
            "largest_abs_order_loss_gap": flt(max_gap["abs_loss_gap"]),
        }
        for family, label in CORR_FAMILIES:
            summary[label] = corr_map.get(family, float("nan"))
        summary_rows.append(summary)

    return summary_rows, config_rows, strategy_rows, method_status_rows


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            "<style>"
            "text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#222}"
            ".small{font-size:10px}.axis{stroke:#333;stroke-width:1}"
            ".grid{stroke:#ddd;stroke-width:1}.legend{font-size:11px}"
            "</style>"
        ),
    ]


def add_text(
    lines: list[str],
    x: float,
    y: float,
    text: str,
    cls: str = "",
    anchor: str = "start",
    rotate: float | None = None,
) -> None:
    attrs = f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"'
    if cls:
        attrs += f' class="{cls}"'
    if rotate is not None:
        attrs += f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
    lines.append(f"<text {attrs}>{escape(str(text))}</text>")


def add_legend(lines: list[str], items: list[tuple[str, str]], x: float, y: float) -> None:
    for idx, (label, color) in enumerate(items):
        yy = y + idx * 18
        lines.append(f'<rect x="{x}" y="{yy - 10}" width="12" height="12" fill="{color}"/>')
        add_text(lines, x + 18, yy, label, "legend")


def save_svg(lines: list[str], path: Path) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def grouped_bar_svg(
    path: Path,
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float], str]],
    y_min: float,
    y_max: float,
    y_label: str,
    log: bool = False,
) -> None:
    width, height = 1250, 560
    ml, mr, mt, mb = 80, 260, 45, 130
    plot_width, plot_height = width - ml - mr, height - mt - mb
    lines = svg_header(width, height)
    add_text(lines, ml, 24, title)

    def scale_y(value: float) -> float:
        if log:
            value = math.log10(max(value, 1e-12))
            low, high = math.log10(y_min), math.log10(y_max)
        else:
            low, high = y_min, y_max
        return mt + (high - value) / (high - low) * plot_height

    if log:
        ticks = [10**x for x in range(math.floor(math.log10(y_min)), math.ceil(math.log10(y_max)) + 1)]
    else:
        ticks = [y_min + (y_max - y_min) * i / 5 for i in range(6)]
    for tick in ticks:
        y = scale_y(tick)
        lines.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_width}" y2="{y:.1f}" class="grid"/>')
        add_text(lines, ml - 8, y + 4, fmt(tick), "small", "end")
    if y_min < 0 < y_max and not log:
        y0 = scale_y(0)
        lines.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml + plot_width}" y2="{y0:.1f}" stroke="#111"/>')
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + plot_height}" class="axis"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + plot_height}" x2="{ml + plot_width}" y2="{mt + plot_height}" class="axis"/>')
    add_text(lines, 18, mt + plot_height / 2, y_label, anchor="middle", rotate=-90)

    group_width = plot_width / len(categories)
    bar_width = min(16, group_width / (len(series) + 1))
    for category_idx, category in enumerate(categories):
        center_x = ml + group_width * category_idx + group_width / 2
        for series_idx, (_name, values, color) in enumerate(series):
            value = values[category_idx]
            x = center_x + (series_idx - (len(series) - 1) / 2) * bar_width - bar_width / 2
            y = scale_y(max(value, y_min) if log else value)
            y0 = scale_y(y_min if log else 0)
            lines.append(
                f'<rect x="{x:.1f}" y="{min(y, y0):.1f}" width="{bar_width * 0.86:.1f}" '
                f'height="{abs(y0 - y):.1f}" fill="{color}"/>'
            )
        add_text(lines, center_x - 5, mt + plot_height + 18, category, "small", "end", rotate=-35)
    add_legend(lines, [(name, color) for name, _values, color in series], ml + plot_width + 25, mt + 20)
    save_svg(lines, path)


def mid_comparison_svg(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    width, height = 1250, 520
    lines = svg_header(width, height)
    add_text(lines, 70, 24, "Mid configuration model-family comparison")
    mid_names = ["70M_mid_3bit_12mods", "160M_mid_3bit_12mods", "Qwen2.5_1.5B_mid_3bit_12mods"]
    mids = [next(row for row in summary_rows if row["run"] == name) for name in mid_names]
    cats = ["rho add.", "rho PPL", "Taylor", "Frob.", "Spec-order"]
    keys = ["rho additivity", "rho PPL", "Taylor vs loss", "Frobenius vs loss", "Spectrum entropy vs order"]
    ml, mt, plot_width, plot_height = 70, 55, 610, 330
    y_min, y_max = -0.1, 0.8

    def scale_y(value: float) -> float:
        return mt + (y_max - value) / (y_max - y_min) * plot_height

    for i in range(6):
        tick = y_min + (y_max - y_min) * i / 5
        y = scale_y(tick)
        lines.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_width}" y2="{y:.1f}" class="grid"/>')
        add_text(lines, ml - 8, y + 4, fmt(tick), "small", "end")
    y0 = scale_y(0)
    lines.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml + plot_width}" y2="{y0:.1f}" stroke="#111"/>')

    group_width, bar_width = plot_width / len(cats), 18
    for category_idx, category in enumerate(cats):
        center_x = ml + group_width * category_idx + group_width / 2
        for series_idx, row in enumerate(mids):
            value = row[keys[category_idx]]
            x = center_x + (series_idx - 1) * bar_width - bar_width / 2
            y = scale_y(value)
            color = COLORS[row["run"]]
            lines.append(
                f'<rect x="{x:.1f}" y="{min(y, y0):.1f}" width="{bar_width * 0.85:.1f}" '
                f'height="{abs(y - y0):.1f}" fill="{color}"/>'
            )
        add_text(lines, center_x, mt + plot_height + 20, category, "small", "middle")
    add_text(lines, ml, mt + plot_height + 50, "Correlation metrics", "small")

    ml2, plot_width2 = 760, 360

    def transform_deg(value: float) -> float:
        return math.log10(1 + max(value, 0) / 10)

    def scale_y2(transformed: float) -> float:
        return mt + (2.8 - transformed) / 2.8 * plot_height

    for tick in [0, 10, 100, 1000, 10000]:
        y = scale_y2(transform_deg(tick))
        lines.append(f'<line x1="{ml2}" y1="{y:.1f}" x2="{ml2 + plot_width2}" y2="{y:.1f}" class="grid"/>')
        add_text(lines, ml2 - 8, y + 4, fmt(tick), "small", "end")

    strat_keys = ["fixed_qsr_naive_ppl", "fixed_qsr_default_ppl", "slim_like_srq_proxy_ppl", "hessian_layerwise_ppl"]
    labels = ["naive QSR", "default QSR", "SLiM proxy", "H-layer"]
    strat_colors = ["#76B7B2", "#9C755F", "#BAB0AC", "#2F4B7C"]
    group_width2, bar_width2 = plot_width2 / len(mids), 17
    y_base = scale_y2(0)
    for category_idx, row in enumerate(mids):
        center_x = ml2 + group_width2 * category_idx + group_width2 / 2
        base_ppl = row["baseline_ppl"]
        for series_idx, key in enumerate(strat_keys):
            degradation = max(0, row[key] - base_ppl)
            y = scale_y2(transform_deg(degradation))
            x = center_x + (series_idx - 1.5) * bar_width2 - bar_width2 / 2
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width2 * 0.85:.1f}" '
                f'height="{y_base - y:.1f}" fill="{strat_colors[series_idx]}"/>'
            )
        label = row["run"].replace("_3bit_12mods", "")
        add_text(lines, center_x, mt + plot_height + 18, label, "small", "middle", rotate=-20)
    add_text(lines, ml2, mt + plot_height + 50, "PPL degradation vs baseline, log-like scale", "small")
    add_legend(lines, [(row["run"], COLORS[row["run"]]) for row in mids] + list(zip(labels, strat_colors)), 70, 430)
    save_svg(lines, path)


def scatter_svg(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    width, height = 1100, 430
    lines = svg_header(width, height)
    add_text(lines, 60, 24, "Representative |rho_H| vs |additivity error|")
    reps = [
        ("70M_strong_2bit_12mods", "Pythia-70M strong"),
        ("Qwen2.5_1.5B_mid_3bit_12mods", "Qwen2.5-1.5B mid"),
    ]
    for panel_idx, (run, title) in enumerate(reps):
        spec = next(item for item in RUNS if item["run"] == run)
        rows = read_csv(RESULTS / spec["dir"] / "metrics" / "additivity.csv")
        xvals = [flt(row["abs_rho_h"]) for row in rows]
        yvals = [flt(row["abs_additivity_error"]) for row in rows]
        ml, mt, plot_width, plot_height = 70 + panel_idx * 530, 55, 410, 285
        xmax = max(xvals) * 1.1
        ymax = max(yvals) * 1.1
        lines.append(f'<rect x="{ml}" y="{mt}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#ccc"/>')
        for i in range(5):
            x = ml + plot_width * i / 4
            y = mt + plot_height * i / 4
            lines.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + plot_height}" class="grid"/>')
            lines.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_width}" y2="{y:.1f}" class="grid"/>')
        for xval, yval in zip(xvals, yvals):
            sx = ml + xval / xmax * plot_width
            sy = mt + plot_height - yval / ymax * plot_height
            lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3.5" fill="{COLORS[run]}" opacity="0.75"/>')
        rho = next(row for row in summary_rows if row["run"] == run)["rho additivity"]
        add_text(lines, ml, mt - 14, title)
        add_text(lines, ml + 8, mt + 18, f"Spearman={rho:.3f}", "small")
        add_text(lines, ml + plot_width / 2, mt + plot_height + 35, "|rho_H|", "small", "middle")
        add_text(lines, ml - 45, mt + plot_height / 2, "|add. err|", "small", "middle", rotate=-90)
        add_text(lines, ml, mt + plot_height + 14, "0", "small")
        add_text(lines, ml + plot_width, mt + plot_height + 14, fmt(xmax), "small", "end")
    save_svg(lines, path)


def write_figures(summary_rows: list[dict[str, Any]], strategy_rows: list[dict[str, Any]]) -> None:
    corr_cats = [
        "rho additivity",
        "rho PPL",
        "rho zero-shot",
        "Taylor vs loss",
        "Frobenius vs loss",
        "Param cosine vs add.",
        "Act recon vs loss",
        "Trace-only vs loss",
        "Spectrum entropy vs order",
    ]
    corr_series = [
        (row["run"], [row[cat] for cat in corr_cats], COLORS[row["run"]])
        for row in summary_rows
    ]
    grouped_bar_svg(
        FIGDIR / "correlation_by_config_model_qwen.svg",
        "Correlation comparison across pretrained runs",
        corr_cats,
        corr_series,
        -0.55,
        0.85,
        "Spearman correlation",
    )

    strategy_order = ["baseline", "fixed_qsr_naive", "fixed_qsr_default", "slim_like_srq_proxy", "hessian_layerwise"]
    labels = ["baseline", "naive QSR", "default QSR", "SLiM-like SRQ", "Hessian layer-wise"]
    strategy_colors = ["#4C78A8", "#76B7B2", "#9C755F", "#BAB0AC", "#2F4B7C"]
    strategy_series = []
    for label, strategy, color in zip(labels, strategy_order, strategy_colors):
        values = []
        for row in summary_rows:
            sub = [item for item in strategy_rows if item["run"] == row["run"] and item["strategy"] == strategy]
            values.append(flt(sub[0]["perplexity"]) if sub else 1.0)
        strategy_series.append((label, values, color))
    grouped_bar_svg(
        FIGDIR / "strategy_ppl_by_config_model_qwen.svg",
        "Strategy PPL comparison",
        [row["run"] for row in summary_rows],
        strategy_series,
        10,
        2e8,
        "Perplexity (log scale)",
        log=True,
    )

    mid_comparison_svg(FIGDIR / "mid_config_model_family_comparison_qwen.svg", summary_rows)
    scatter_svg(FIGDIR / "additivity_scatter_representative_qwen.svg", summary_rows)


def write_markdown(summary_rows: list[dict[str, Any]], config_rows: list[dict[str, Any]]) -> None:
    qwen = next(row for row in summary_rows if row["run"] == "Qwen2.5_1.5B_mid_3bit_12mods")
    py160 = next(row for row in summary_rows if row["run"] == "160M_mid_3bit_12mods")
    py70_mid = next(row for row in summary_rows if row["run"] == "70M_mid_3bit_12mods")
    py70_strong = next(row for row in summary_rows if row["run"] == "70M_strong_2bit_12mods")
    models = [
        "/home/wangmeiqi/ZHuan/model/Qwen2.5-0.5B",
        "/home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B (used for this validation)",
        "/home/wangmeiqi/ZHuan/model/Qwen3-0.6B",
        "/home/spco/base-2-bitnet/.hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct",
        "/home/wangmeiqi/zjh/meta-llama/Llama-2-7b-hf",
        "Additional vision-language candidates exist, e.g. Qwen2.5-VL/LLaVA/Cambrian, but the current script is text-only.",
    ]
    model_note = [
        "# 210 Model Search Operator Note",
        "",
        "This note records the model paths found during the successful 210 experiment stage. It is not a fresh reproducible scan log from the metrics bundle.",
        "A later read-only re-search attempt was blocked by the 210 login shell's `nvm`/`PREFIX` initialization error before the remote search command could produce output.",
        "",
    ]
    model_note.extend(f"- `{model}`" for model in models)
    (OUT / "model_search_210_operator_note.md").write_text("\n".join(model_note) + "\n", encoding="utf-8")

    md = """# Pretrained LLM Orthogonality Conclusion With Qwen Validation

This summary updates the model-scale conclusion with the 210-server Qwen2.5-1.5B run. It is intentionally framed as an evidence audit for the original goal: the claim is not that a landscape can be drawn, but whether Hessian cross-terms explain complementarity, conflict, and non-commutative compression order.

Scope notes:
- `rho_H` is computed with a layer-local Hessian/Gauss-Newton proxy from activation covariance `X^T X`; it is not the exact full-model Hessian.
- Additivity rows use linearized perturbation sums `W + Delta_i + Delta_j`. Executable order gaps use actual sequential application and are reported separately.
- Pythia runs use 1016 PPL tokens and 8 zero-shot examples/task. Qwen2.5-1.5B uses 508 PPL tokens and 4 zero-shot examples/task, so cross-family numbers should be read qualitatively rather than as a strict leaderboard.
- These summarized runs were generated before the zero-shot-backup text loader was corrected to interleave tasks. Their metadata records candidate backup availability for `arc_easy,hellaswag`, but PPL/calibration text selection may be dominated by the first listed task. The runner now records actual task counts for future reruns.
- External GPTQ/AWQ/SparseGPT packages were unavailable in the tested environments. Native coverage is RTN, magnitude, Wanda-style activation-aware pruning, vanilla SVD, and activation-whitened SVD proxy. `slim_like_srq_proxy` is a fixed recipe proxy, not the official SLiM implementation.

## 210 Model Search

The successful 210 experiment stage found these relevant text-model candidates. This is an operator note, not a reproducible scan artifact from the metrics bundle:
"""
    for model in models:
        md += f"- `{model}`\n"
    md += (
        "\nA later read-only re-search attempt in this turn was blocked before command execution by the 210 login "
        "shell's `nvm`/`PREFIX` initialization error, so the final list above is based on the already completed "
        "210 experiment-stage discovery and the Qwen run artifact path. The same note is saved as "
        "`model_search_210_operator_note.md`.\n\n"
    )
    md += "## Experiment Configurations\n\n"
    md += md_table(
        config_rows,
        [
            "run",
            "model",
            "param_label",
            "server_source",
            "selected_layers",
            "bits",
            "keep_fraction",
            "rank_fraction",
            "q_method",
            "s_method",
            "r_method",
            "eval_tokens",
            "zero_shot_examples_per_task",
            "text_source_used",
        ],
        [
            "Run",
            "Model",
            "Params",
            "Server/source",
            "Modules",
            "Bits",
            "Keep",
            "Rank",
            "Q",
            "S",
            "R",
            "PPL tokens",
            "Zero-shot examples",
            "Text source",
        ],
    )
    md += "\n\n### Selected Layer Names\n\n"
    for row in config_rows:
        md += f"- `{row['run']}`: {row['selected_layer_names']}\n"

    md += "\n## Cross-Run Result Summary\n\n"
    md += md_table(
        summary_rows,
        [
            "run",
            "rho additivity",
            "rho PPL",
            "rho zero-shot",
            "Taylor vs loss",
            "Frobenius vs loss",
            "Trace-only vs loss",
            "Spectrum entropy vs order",
            "baseline_ppl",
            "hessian_layerwise_ppl",
            "fixed_qsr_naive_ppl",
            "fixed_qsr_default_ppl",
            "slim_like_srq_proxy_ppl",
            "hessian_vs_fixed_naive_ppl_delta",
            "hessian_vs_fixed_default_ppl_delta",
            "hessian_vs_slim_proxy_ppl_delta",
        ],
        [
            "Run",
            "rho add.",
            "rho PPL",
            "rho zero-shot",
            "Taylor-loss",
            "Frob-loss",
            "Trace-loss",
            "Spectrum-order",
            "Base PPL",
            "Hessian PPL",
            "Naive QSR PPL",
            "Default QSR PPL",
            "SLiM-proxy PPL",
            "Hessian-Naive",
            "Hessian-Fixed",
            "Hessian-SLiM",
        ],
    )
    md += "\n\n## MVP Criteria Check\n\n"
    md += (
        "1. Hessian cosine heatmaps exist for q/s/r on all pretrained runs, including Qwen2.5-1.5B at "
        f"`{RUNS[-1]['dir']}/figures/hessian_cosine_heatmap.png`.\n"
    )
    md += (
        "2. High `rho_H` predicting additivity is supported in the stress/Qwen settings but not uniformly. "
        f"Pythia-70M strong gives Spearman {fmt(py70_strong['rho additivity'])}; "
        f"Qwen2.5-1.5B gives {fmt(qwen['rho additivity'])}; "
        f"Pythia-160M mid is weak at {fmt(py160['rho additivity'])}.\n"
    )
    md += (
        "3. Order non-commutativity is observable. Qwen's largest order-gap row is "
        f"`{qwen['largest_order_gap_layer']}` with `{qwen['largest_order_gap_orders']}` and absolute loss gap "
        f"{fmt(qwen['largest_abs_order_loss_gap'])}. Across settings, singular-spectrum shifts explain order gaps "
        f"more consistently than symmetric Hessian overlap: Qwen spectrum-order {fmt(qwen['Spectrum entropy vs order'])}; "
        f"Pythia-160M spectrum-order {fmt(py160['Spectrum entropy vs order'])}.\n"
    )
    md += (
        "4. Hessian-guided layer-wise selection beats both true naive QSR and default fixed QSR in every summarized run, including "
        f"Qwen where PPL is {fmt(qwen['hessian_layerwise_ppl'])} vs naive QSR {fmt(qwen['fixed_qsr_naive_ppl'])} "
        f"and default QSR {fmt(qwen['fixed_qsr_default_ppl'])}. It does not always beat the SLiM-like fixed recipe proxy: "
        f"Qwen is {fmt(qwen['hessian_layerwise_ppl'])} vs SLiM-proxy {fmt(qwen['slim_like_srq_proxy_ppl'])}.\n"
    )

    md += "\n## Method Effectiveness Analysis\n\n"
    md += (
        "The strongest evidence for the framework is the Taylor/cross-term diagnostic rather than raw `rho_H` alone. "
        f"In matched Pythia mid settings, Taylor-vs-loss is {fmt(py70_mid['Taylor vs loss'])} for 70M and "
        f"{fmt(py160['Taylor vs loss'])} for 160M, both above the Frobenius baseline "
        f"({fmt(py70_mid['Frobenius vs loss'])} and {fmt(py160['Frobenius vs loss'])}). In the larger Qwen run, "
        f"Taylor-vs-loss remains useful at {fmt(qwen['Taylor vs loss'])}, above Frobenius "
        f"{fmt(qwen['Frobenius vs loss'])}, and also above/near trace-only {fmt(qwen['Trace-only vs loss'])}.\n\n"
    )
    md += (
        "Raw `rho_H` is a partial predictor. It tracks linearized additivity under stronger perturbation and the "
        f"Qwen family check ({fmt(py70_strong['rho additivity'])} and {fmt(qwen['rho additivity'])}), but it is weak "
        f"on Pythia-160M mid ({fmt(py160['rho additivity'])}) and only weakly connected to real PPL/zero-shot "
        f"degradation in Qwen ({fmt(qwen['rho PPL'])}, {fmt(qwen['rho zero-shot'])}). The conclusion should therefore be: "
        "Hessian cross-terms are useful as a diagnostic feature and selector input, not a universal scalar predictor by themselves.\n\n"
    )
    md += (
        "The layer-wise selector is promising but not settled. It consistently improves over true naive Q->S->R and "
        "the default fixed Q->S->R baseline under the same compression budget. Against a stronger SLiM-like fixed recipe, it wins on the Pythia "
        f"runs, narrowly on 160M ({fmt(py160['hessian_layerwise_ppl'])} vs {fmt(py160['slim_like_srq_proxy_ppl'])}), "
        f"but loses slightly on Qwen ({fmt(qwen['hessian_layerwise_ppl'])} vs {fmt(qwen['slim_like_srq_proxy_ppl'])}). "
        "That means the paper claim should be phrased as evidence that Hessian-guided selection can avoid bad fixed "
        "orders and sometimes improve over strong recipes, while a broader selector search is still needed for a universal advantage.\n"
    )

    md += "\n## Comparison With Existing Work\n\n"
    md += (
        "Existing compression work such as SLiM-like fixed recipes, QSLR-style quantization/low-rank combinations, "
        "LoSparse-style low-rank+sparse decomposition, and LQ-LoRA-style low-rank/quantized adaptation already covers "
        "much of the algorithmic combination space. The distinct contribution supported here is not 'Q+S+R works', "
        "but a measurement layer that asks which pairs conflict or complement in a layer and why the order changes the outcome.\n\n"
    )
    md += (
        "Compared with those fixed or algorithm-specific recipes, this framework exposes: pairwise Hessian overlap "
        "heatmaps for Q/S/R perturbations; linearized additivity error tied to cross-terms; executable order gaps tied "
        "partly to singular-spectrum shifts; and a layer-wise choice rule. The current evidence is enough to motivate "
        "the framework, but not enough to claim dominance over official SLiM, GPTQ/AWQ, or SparseGPT implementations "
        "because those packages were not run here.\n"
    )
    md += "\n## Figures\n\n"
    for filename in [
        "correlation_by_config_model_qwen.svg",
        "strategy_ppl_by_config_model_qwen.svg",
        "mid_config_model_family_comparison_qwen.svg",
        "additivity_scatter_representative_qwen.svg",
    ]:
        md += f"- `figures/{filename}`\n"
    md += (
        "\n## Files\n\n"
        "- `summary.csv`: machine-readable cross-run metrics.\n"
        "- `experiment_configurations.csv`: exact setup details, selected layers, data source fingerprints, and evaluation budgets.\n"
        "- `strategy_rows.csv`: raw strategy-level PPL/accuracy rows.\n"
        "- `method_status_all_runs.csv`: available/native/proxy/unavailable method status for Q/GPTQ/AWQ, S/SparseGPT, and R/SVD-LLM-style methods.\n"
        "- `model_search_210_operator_note.md`: operator note for the 210 model paths used to choose the Qwen validation target.\n"
    )
    (OUT / "summary.md").write_text(md, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    summary_rows, config_rows, strategy_rows, method_status_rows = collect_rows()

    write_csv(OUT / "summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(OUT / "experiment_configurations.csv", config_rows, list(config_rows[0]))
    write_csv(OUT / "strategy_rows.csv", strategy_rows, sorted({key for row in strategy_rows for key in row}))
    write_csv(OUT / "method_status_all_runs.csv", method_status_rows, sorted({key for row in method_status_rows for key in row}))
    write_figures(summary_rows, strategy_rows)
    write_markdown(summary_rows, config_rows)
    print(OUT)


if __name__ == "__main__":
    main()
