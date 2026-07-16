#!/usr/bin/env python3
"""Plot the verified three-job scalability smoke from generated CSVs only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "paper" / "results"
ENDPOINTS = RESULT_DIR / "scaling_pilot_endpoints.csv"
PAIRS = RESULT_DIR / "scaling_pilot_pairs.csv"
MODELS = RESULT_DIR / "scaling_pilot_models.csv"
PATHS = RESULT_DIR / "scaling_pilot_paths.csv"
REPORT_MANIFEST = RESULT_DIR / "scaling_pilot_manifest.json"

REPORT_SCHEMA = "scaling_pilot_report.v1"
FIGURE_SCHEMA = "scaling_pilot_figure.v1"
EVIDENCE_STATUS = "verified_single_seed_scalability_smoke"
EVIDENCE_ROLE = "scalability_smoke"
INTERVAL_SEMANTICS = (
    "fixed-window descriptive mean +/- 1.96 sample standard errors; "
    "not an independence-based confidence interval or significance test"
)
MODEL_ORDER = (
    "pythia70m_full_mlp_pilot",
    "opt125m_depth_mlp_pilot",
    "qwen3_06b_depth_mlp_pilot",
)
MODEL_CONTRACT = {
    "pythia70m_full_mlp_pilot": ("Pythia-70M", "full_mlp_weights", 12),
    "opt125m_depth_mlp_pilot": ("OPT-125M", "five_depth_mlp_weights", 10),
    "qwen3_06b_depth_mlp_pilot": ("Qwen3-0.6B", "five_depth_mlp_weights", 10),
}
EXPECTED_STRATEGIES = (
    "Q",
    "Q_global_scale",
    "Q_block_scale",
    "Q+S",
    "Q+S_OBS",
    "Q+L",
    "Q+S+L_QL_budget",
    "Q+S+L_QL_budget_component_scale",
    "Q+S+L",
    "Q+S_OBS+L",
    "Q+S+L_component_scale",
)
EXPECTED_COMPARISONS = {
    "global_scale_vs_q": ("Q_global_scale", "Q"),
    "block_scale_vs_q": ("Q_block_scale", "Q"),
    "obs_vs_qs": ("Q+S_OBS", "Q+S"),
    "strict_qsl_vs_ql": ("Q+S+L_QL_budget_component_scale", "Q+L"),
}
EXPECTED_PATH_STRATEGIES = (
    "Q",
    "Q_block_scale",
    "Q+S_OBS",
    "Q+L",
    "Q+S+L_QL_budget_component_scale",
    "Q+S+L_component_scale",
)
EXPECTED_EPSILONS = (
    0.0,
    0.03125,
    0.0625,
    0.09375,
    0.125,
    0.1875,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
)

STRICT = "Q+S+L_QL_budget_component_scale"
QL = "Q+L"
MODEL_COLORS = ("#0072B2", "#009E73", "#D55E00")
RHO_COLORS = {"rho_sl": "#0072B2", "rho_qs": "#E69F00", "rho_ql": "#CC79A7"}
METHODS = (
    ("Q_global_scale", "global scale", "o"),
    ("Q_block_scale", "block scale", "s"),
    ("Q+S", "Q+S", "^"),
    ("Q+S_OBS", "Q+S (OBS)", "v"),
    (QL, "Q+L", "*"),
    (STRICT, "strict scaled QSL", "D"),
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite in {row}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth(value: str, field: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{field} must be true or false, got {value!r}")


def _close(value: float, expected: float, field: str, tolerance: float = 1e-12) -> None:
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ValueError(f"{field} changed: {value!r} != {expected!r}")


def _validated_inputs() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, Any],
]:
    report = json.loads(REPORT_MANIFEST.read_text(encoding="utf-8"))
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("scaling report schema changed")
    if report.get("evidence_status") != EVIDENCE_STATUS:
        raise ValueError("scaling report is not verified single-seed smoke evidence")
    if report.get("observation_count") != 3 or report.get("seed") != 17:
        raise ValueError("scaling report observation/seed contract changed")
    _close(
        float(report.get("target_selected_weight_artifact_ratio")),
        0.258,
        "report target selected-weight rate",
    )
    if tuple(report.get("strategy_order", ())) != EXPECTED_STRATEGIES:
        raise ValueError("scaling report strategy order changed")
    if tuple(report.get("comparison_order", ())) != tuple(EXPECTED_COMPARISONS):
        raise ValueError("scaling report comparison order changed")
    if report.get("interval_semantics") != INTERVAL_SEMANTICS:
        raise ValueError("scaling report interval semantics changed")

    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("scaling report outputs must be a SHA mapping")
    for path in (ENDPOINTS, PAIRS, MODELS, PATHS):
        expected = outputs.get(path.name)
        if not isinstance(expected, str) or _sha(path) != expected:
            raise ValueError(f"{path.name} does not match scaling report manifest")

    endpoint_rows = _read(ENDPOINTS)
    pair_rows = _read(PAIRS)
    model_rows = _read(MODELS)
    path_rows = _read(PATHS)

    if tuple(row["model_id"] for row in model_rows) != MODEL_ORDER:
        raise ValueError("model row order/identity changed")
    for row in model_rows:
        model_id = row["model_id"]
        label, scope_id, tensor_count = MODEL_CONTRACT[model_id]
        if (
            row["model"] != label
            or row["scope_id"] != scope_id
            or int(row["selected_tensors"]) != tensor_count
            or row["evidence_role"] != EVIDENCE_ROLE
            or _truth(row["seed_aggregation_allowed"], "seed aggregation")
        ):
            raise ValueError(f"{model_id}: model/scope/evidence contract changed")
        if int(row["eval_tokens"]) != 1016:
            raise ValueError(f"{model_id}: evaluation token count changed")

    if len(endpoint_rows) != 33:
        raise ValueError(f"expected 33 endpoint rows, got {len(endpoint_rows)}")
    for model_id in MODEL_ORDER:
        rows = [row for row in endpoint_rows if row["model_id"] == model_id]
        if tuple(row["strategy"] for row in rows) != EXPECTED_STRATEGIES:
            raise ValueError(f"{model_id}: endpoint strategy order/set changed")
        for row in rows:
            if int(row["seed"]) != 17 or int(row["eval_tokens"]) != 1016:
                raise ValueError(f"{model_id}: endpoint seed/token contract changed")
            _close(
                float(row["target_selected_weight_rate"]),
                0.258,
                f"{model_id}: endpoint target rate",
            )

    if len(pair_rows) != 12:
        raise ValueError(f"expected 12 paired rows, got {len(pair_rows)}")
    for model_id in MODEL_ORDER:
        rows = [row for row in pair_rows if row["model_id"] == model_id]
        if tuple(row["comparison_id"] for row in rows) != tuple(EXPECTED_COMPARISONS):
            raise ValueError(f"{model_id}: paired comparison order/set changed")
        for row in rows:
            left, right = EXPECTED_COMPARISONS[row["comparison_id"]]
            if row["left_strategy"] != left or row["right_strategy"] != right:
                raise ValueError(f"{model_id}: paired comparison direction changed")
            if int(row["fixed_window_count"]) != 8:
                raise ValueError(f"{model_id}: fixed-window count changed")
            if row["interval_semantics"] != INTERVAL_SEMANTICS:
                raise ValueError(f"{model_id}: paired interval semantics changed")
            if row["comparison_id"] == "strict_qsl_vs_ql" and not _truth(
                row["same_final_file_bytes"], "strict same-file-bytes"
            ):
                raise ValueError(f"{model_id}: strict scaled QSL is not byte-equal to Q+L")

    if len(path_rows) != 234:
        raise ValueError(f"expected 234 loss-path rows, got {len(path_rows)}")
    for model_id in MODEL_ORDER:
        for strategy in EXPECTED_PATH_STRATEGIES:
            rows = [
                row
                for row in path_rows
                if row["model_id"] == model_id and row["strategy"] == strategy
            ]
            epsilons = tuple(float(row["epsilon"]) for row in rows)
            if epsilons != EXPECTED_EPSILONS:
                raise ValueError(f"{model_id}/{strategy}: epsilon grid changed")
            for row, epsilon in zip(rows, epsilons):
                if row["evidence_role"] != EVIDENCE_ROLE:
                    raise ValueError(f"{model_id}/{strategy}: evidence role changed")
                if int(row["eval_tokens"]) != 1016:
                    raise ValueError(f"{model_id}/{strategy}: eval tokens changed")
                _close(
                    float(row["target_selected_weight_rate"]),
                    0.258,
                    f"{model_id}/{strategy}: path target rate",
                )
                _close(
                    float(row["small_epsilon_fit_max"]),
                    0.125,
                    f"{model_id}/{strategy}: Taylor fit boundary",
                )
                if _truth(row["inside_taylor_fit_interval"], "inside-fit flag") != (
                    epsilon <= 0.125
                ):
                    raise ValueError(f"{model_id}/{strategy}: inside-fit flag changed")
                if _truth(row["fit_is_extrapolation"], "extrapolation flag") != (
                    epsilon > 0.125
                ):
                    raise ValueError(f"{model_id}/{strategy}: extrapolation flag changed")
                if _truth(row["deployable"], "deployable flag") != (epsilon == 1.0):
                    raise ValueError(f"{model_id}/{strategy}: deployable endpoint changed")
    return endpoint_rows, pair_rows, model_rows, path_rows, report


def _style() -> None:
    font_paths = sorted((ROOT / "paper" / "fonts").glob("texgyretermes-*.otf"))
    if len(font_paths) != 4:
        raise FileNotFoundError("expected four repository TeX Gyre Termes font files")
    for path in font_paths:
        font_manager.fontManager.addfont(path)
    family = font_manager.FontProperties(
        fname=ROOT / "paper" / "fonts" / "texgyretermes-regular.otf"
    ).get_name()
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "com-compression-scaling-pilot-v1",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _save(
    fig: plt.Figure,
    stem: Path,
    inputs: list[Path],
    scope: str,
    report: dict[str, Any],
) -> None:
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    fig.savefig(
        pdf_path,
        dpi=300,
        metadata={"Creator": "scaling_pilot_plots.py", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        svg_path,
        dpi=300,
        metadata={"Creator": "scaling_pilot_plots.py", "Date": None},
    )
    fig.savefig(png_path, dpi=300, metadata={"Software": "scaling_pilot_plots.py"})
    output_hashes = {
        path.name: _sha(path) for path in (pdf_path, svg_path, png_path)
    }
    manifest = {
        "schema_version": FIGURE_SCHEMA,
        "evidence_status": report["evidence_status"],
        "scope": scope,
        "inputs": {
            path.relative_to(ROOT).as_posix(): _sha(path)
            for path in inputs
        },
        "plot_script": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha(Path(__file__).resolve()),
        },
        "outputs": output_hashes,
        "interval_semantics": INTERVAL_SEMANTICS,
        "taylor_fit_semantics": (
            "fits use epsilon <= 0.125; dotted curves beyond 0.125 are extrapolations"
        ),
        "prohibited_inference": [
            "no cross-model absolute NLL or perplexity ranking",
            "no whole-model compression claim",
            "no multi-seed or population-significance claim",
            "fixed-window intervals are descriptive and are not significance tests",
            "path correlations apply only to the shown one-dimensional 13-point slices",
        ],
    }
    manifest_path = stem.with_suffix(".manifest.json")
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temporary, manifest_path)


def _model_order(model_rows: list[dict[str, str]]) -> tuple[list[str], dict[str, str]]:
    if len(model_rows) != 3:
        raise ValueError(f"expected three model rows, got {len(model_rows)}")
    order = [row["model_id"] for row in model_rows]
    if len(set(order)) != 3:
        raise ValueError("model ids must be unique")
    return order, {row["model_id"]: row["model"] for row in model_rows}


def efficiency_geometry(
    endpoint_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    model_rows: list[dict[str, str]],
    report: dict[str, Any],
) -> None:
    order, labels = _model_order(model_rows)
    by_endpoint = {(row["model_id"], row["strategy"]): row for row in endpoint_rows}
    by_pair = {(row["model_id"], row["comparison_id"]): row for row in pair_rows}
    if len(by_endpoint) != 33:
        raise ValueError("expected 33 unique endpoint rows")

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.72), constrained_layout=True)

    ax = axes[0]
    for model_index, model_id in enumerate(order):
        color = MODEL_COLORS[model_index]
        for strategy, _, marker in METHODS:
            row = by_endpoint[(model_id, strategy)]
            x = _finite(row, "added_physical_bits_per_selected_parameter_vs_q")
            y = _finite(row, "nll_recovery_vs_q")
            face = "none" if strategy == STRICT else color
            size = 54 if marker == "*" else 30
            ax.scatter(
                x,
                y,
                marker=marker,
                s=size,
                facecolors=face,
                edgecolors=color,
                linewidths=1.05,
                zorder=4,
            )
    ax.axhline(0.0, color="#555555", linewidth=0.7)
    ax.axvline(0.0, color="#999999", linewidth=0.6)
    ax.set_xlabel("Added exact bits / selected parameter")
    ax.set_ylabel("NLL recovery from Q (nats/token)")
    ax.grid(linestyle=":", linewidth=0.45, alpha=0.45)
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=labels[mid])
        for mid, color in zip(order, MODEL_COLORS)
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="#333333",
            markerfacecolor="none" if strategy == STRICT else "#333333",
            label=label,
        )
        for strategy, label, marker in METHODS
    ]
    first = ax.legend(
        handles=model_handles,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.01),
        ncol=3,
        handletextpad=0.2,
        columnspacing=0.45,
    )
    ax.add_artist(first)
    ax.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        handletextpad=0.2,
        columnspacing=0.55,
    )
    ax.text(
        0.04,
        0.96,
        "within-model only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.text(-0.18, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    x_positions = list(range(3))
    for x, (model_id, color) in enumerate(zip(order, MODEL_COLORS)):
        row = by_pair[(model_id, "strict_qsl_vs_ql")]
        mean = _finite(row, "fixed_window_mean_nll_difference_left_minus_right")
        low = _finite(row, "fixed_window_descriptive_interval_low")
        high = _finite(row, "fixed_window_descriptive_interval_high")
        ax.errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            color=color,
            markerfacecolor="none",
            markeredgewidth=1.15,
            markersize=5.2,
            capsize=2.5,
            linewidth=1.0,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.75)
    ax.set_xticks(x_positions, [labels[mid] for mid in order], rotation=18, ha="right")
    ax.set_ylabel("Strict scaled QSL $-$ QL\npaired NLL (nats/token)")
    ax.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.45)
    ax.text(
        0.98,
        0.98,
        "8 fixed windows; descriptive",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    ax.text(-0.18, 1.03, "(b)", transform=ax.transAxes, fontweight="bold")

    ax = axes[2]
    width = 0.22
    model_by_id = {row["model_id"]: row for row in model_rows}
    for rho_index, (field, color, label) in enumerate(
        (
            ("strict_rho_sl", RHO_COLORS["rho_sl"], r"$\rho_{SL}$"),
            ("strict_rho_qs", RHO_COLORS["rho_qs"], r"$\rho_{QS}$"),
            ("strict_rho_ql", RHO_COLORS["rho_ql"], r"$\rho_{QL}$"),
        )
    ):
        values = [_finite(model_by_id[mid], field) for mid in order]
        xs = [x + (rho_index - 1) * width for x in x_positions]
        ax.bar(xs, values, width=width, color=color, label=label, alpha=0.88)
    ax.axhspan(-0.1, 0.1, color="#BBBBBB", alpha=0.25, label=r"$|\rho|\leq0.1$")
    ax.axhline(0.0, color="#333333", linewidth=0.7)
    ax.set_xticks(x_positions, [labels[mid] for mid in order], rotation=18, ha="right")
    ax.set_ylabel("Signed PSD-proxy correlation")
    ax.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.45)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.01),
        ncol=2,
        columnspacing=0.55,
        handletextpad=0.25,
    )
    ax.text(-0.18, 1.03, "(c)", transform=ax.transAxes, fontweight="bold")

    stem = FIGURE_DIR / "scaling_pilot_efficiency_geometry"
    _save(
        fig,
        stem,
        [REPORT_MANIFEST, ENDPOINTS, PAIRS, MODELS],
        (
            "three separate single-seed, single-rate scalability-smoke jobs; "
            "conservative strict scaled QSL is tail-padded to Q+L final bytes; "
            "natural underfill is 31,360/640/1,024 bytes for Pythia/OPT/Qwen; "
            "models have unequal tensor scopes"
        ),
        report,
    )
    plt.close(fig)


def loss_paths(
    path_rows: list[dict[str, str]],
    model_rows: list[dict[str, str]],
    report: dict[str, Any],
) -> None:
    order, labels = _model_order(model_rows)
    wanted = {QL, STRICT}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in path_rows:
        if row["strategy"] in wanted:
            grouped.setdefault((row["model_id"], row["strategy"]), []).append(row)
    if set(grouped) != {(model_id, strategy) for model_id in order for strategy in wanted}:
        raise ValueError("missing QL or strict-QSL path")

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.60), constrained_layout=True)
    style = {
        QL: ("#0072B2", "o", "Q+L"),
        STRICT: ("#D55E00", "D", "strict scaled QSL"),
    }
    for panel, model_id in enumerate(order):
        ax = axes[panel]
        fit_max = None
        annotations: list[str] = []
        for strategy in (QL, STRICT):
            rows = sorted(grouped[(model_id, strategy)], key=lambda row: float(row["epsilon"]))
            if len(rows) != 13:
                raise ValueError(f"{model_id}/{strategy} must have 13 path points")
            eps = [_finite(row, "epsilon") for row in rows]
            measured = [_finite(row, "nll_delta") for row in rows]
            fitted = [_finite(row, "taylor_fit_nll_delta") for row in rows]
            current_fit_max = _finite(rows[0], "small_epsilon_fit_max")
            fit_max = current_fit_max if fit_max is None else fit_max
            if abs(fit_max - current_fit_max) > 1e-12:
                raise ValueError("fit boundary differs within a model")
            color, marker, label = style[strategy]
            ax.plot(eps, measured, color=color, marker=marker, markersize=3.2, linewidth=1.1, label=label)
            inside = [index for index, epsilon in enumerate(eps) if epsilon <= current_fit_max]
            outside = [inside[-1]] + [
                index for index, epsilon in enumerate(eps) if epsilon > current_fit_max
            ]
            ax.plot(
                [eps[index] for index in inside],
                [fitted[index] for index in inside],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
            )
            ax.plot(
                [eps[index] for index in outside],
                [fitted[index] for index in outside],
                color=color,
                linestyle=":",
                linewidth=0.95,
                alpha=0.55,
            )
            correlation = _finite(rows[0], "hessian_proxy_nll_correlation")
            short_label = "QL" if strategy == QL else "scaled QSL"
            annotations.append(f"{short_label}: path r={correlation:.3f}")
        assert fit_max is not None
        ax.axvspan(0.0, fit_max, color="#BBBBBB", alpha=0.22)
        ax.axvline(fit_max, color="#777777", linewidth=0.65, linestyle=":")
        ax.axhline(0.0, color="#444444", linewidth=0.65)
        ax.set_title(labels[model_id])
        ax.set_xlabel(r"Path scale $\epsilon$")
        if panel == 0:
            ax.set_ylabel(r"Held-out $\Delta$NLL (nats/token)")
        ax.grid(linestyle=":", linewidth=0.42, alpha=0.4)
        ax.text(
            0.04,
            0.96,
            "\n".join(annotations),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        ax.text(-0.16, 1.04, f"({chr(ord('a') + panel)})", transform=ax.transAxes, fontweight="bold")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=style[QL][0], marker=style[QL][1], label=style[QL][2]),
            Line2D(
                [0], [0], color=style[STRICT][0], marker=style[STRICT][1], label=style[STRICT][2]
            ),
            Line2D([0], [0], color="#555555", linestyle="--", label=r"fit: $\epsilon\leq0.125$"),
            Line2D([0], [0], color="#777777", linestyle=":", label="fit extrapolation"),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.01),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.55,
    )
    stem = FIGURE_DIR / "scaling_pilot_loss_paths"
    _save(
        fig,
        stem,
        [REPORT_MANIFEST, PATHS, MODELS],
        (
            "13-point Q+L and strict-scaled-QSL one-dimensional path slices for three "
            "separate single-seed scalability-smoke jobs"
        ),
        report,
    )
    plt.close(fig)


def main() -> None:
    _style()
    endpoint_rows, pair_rows, model_rows, path_rows, report = _validated_inputs()
    efficiency_geometry(endpoint_rows, pair_rows, model_rows, report)
    loss_paths(path_rows, model_rows, report)
    print(FIGURE_DIR / "scaling_pilot_efficiency_geometry.pdf")
    print(FIGURE_DIR / "scaling_pilot_loss_paths.pdf")


if __name__ == "__main__":
    main()
