from __future__ import annotations

from pathlib import Path
from statistics import mean


def _safe_mean(rows: list[dict[str, object]], key: str) -> float | None:
    vals: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if value == value:
            vals.append(value)
    return mean(vals) if vals else None


def write_markdown_report(
    rows: list[dict[str, object]],
    dynamic_rows: list[dict[str, object]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    alpha_mean = _safe_mean(rows, "alpha")
    pr_mean = _safe_mean(rows, "participation_ratio")
    eff_mean = _safe_mean(rows, "effective_rank")
    trained = [row for row in rows if row.get("variant") == "pretrained"]
    random = [row for row in rows if row.get("variant") == "random"]
    attn = [row for row in rows if row.get("site") == "attn_out"]
    mlp = [row for row in rows if row.get("site") == "mlp_out"]
    resid = [row for row in rows if row.get("site") == "resid_post"]

    trained_alpha = _safe_mean(trained, "alpha")
    random_alpha = _safe_mean(random, "alpha")
    attn_eff = _safe_mean(attn, "effective_rank")
    mlp_eff = _safe_mean(mlp, "effective_rank")
    resid_eff = _safe_mean(resid, "effective_rank")
    dyn_corr = _safe_mean(dynamic_rows, "mean_abs_pc_autocorr")

    lines = [
        "# LLM Spectral Dynamics Report",
        "",
        "This report is generated from the current experiment outputs. Interpret conclusions as preliminary until the full configured model and dataset sweep has completed.",
        "",
        "## Long-tail spectra",
        f"- Mean fitted alpha: {alpha_mean:.4g}" if alpha_mean is not None else "- Mean fitted alpha: unavailable.",
        f"- Mean participation ratio: {pr_mean:.4g}" if pr_mean is not None else "- Mean participation ratio: unavailable.",
        f"- Mean effective rank: {eff_mean:.4g}" if eff_mean is not None else "- Mean effective rank: unavailable.",
        "",
        "## Pretrained vs random-init",
    ]
    if trained_alpha is not None and random_alpha is not None:
        lines.append(f"- Pretrained mean alpha: {trained_alpha:.4g}; random-init mean alpha: {random_alpha:.4g}; delta: {trained_alpha - random_alpha:.4g}.")
    else:
        lines.append("- Comparison unavailable because one variant is missing.")
    lines.extend(["", "## Attention vs FFN"])
    if attn_eff is not None and mlp_eff is not None:
        lines.append(f"- Attention effective rank mean: {attn_eff:.4g}; FFN effective rank mean: {mlp_eff:.4g}.")
    else:
        lines.append("- Attention/FFN comparison unavailable.")
    if resid_eff is not None:
        lines.append(f"- Residual stream effective rank mean: {resid_eff:.4g}.")
    lines.extend(["", "## KV cache"])
    lines.append("- KV-cache spectral and compression results are written separately when `scripts/run_kv_spectra.sh` or KV sites are enabled.")
    lines.extend(["", "## Token-time dynamics"])
    if dyn_corr is not None:
        lines.append(f"- Mean absolute PC autocorrelation across reported lags: {dyn_corr:.4g}. DMD summaries are included in `results/metrics/dynamic_metrics.csv`.")
    else:
        lines.append("- Dynamic metrics unavailable.")
    lines.extend(["", "## Loss and metric associations"])
    loss_mean = _safe_mean(rows, "nll_mean")
    if loss_mean is not None:
        lines.append(f"- Mean token negative log likelihood over analyzed batches: {loss_mean:.4g}. Use the metrics table to correlate alpha, rank metrics, and loss at model/layer/site granularity.")
    else:
        lines.append("- Loss statistics unavailable.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

