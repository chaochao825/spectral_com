#!/usr/bin/env python3
"""Generate ICML table fragments from verified committed result files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SERIALIZED = ROOT / "results" / "pretrained_hessian_repair_pythia70m_serialized_20260714"
ENDPOINTS = SERIALIZED / "strategy_endpoints.csv"
PAIRS = SERIALIZED / "paired_method_comparisons.csv"
COMFORT = SERIALIZED / "comfort_summary.csv"
RUN_CONFIG = SERIALIZED / "run_config.json"

ENDPOINT_ORDER = [
    ("Q", "Q"),
    ("Q_block_scale", r"Q + block scale"),
    ("Q+S", r"Q+S"),
    ("Q+S_OBS", r"Q+S (OBS)"),
    ("Q+L", r"\textbf{Q+L}"),
    ("Q+S+L_QL_budget_component_scale", r"Q+S+L, strict"),
    ("Q+S+L_component_scale", r"Q+S+L, +5{,}056 B"),
]

PAIR_ORDER = [
    ("obs_vs_qs", r"OBS vs. Q+S"),
    ("block_scale_vs_q", r"Block scale vs. Q"),
    ("constrained_scaled_qsl_vs_ql", r"Strict QSL vs. Q+L"),
    ("unconstrained_scaled_qsl_vs_ql", r"QSL (+5{,}056 B) vs. Q+L"),
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def endpoint_table() -> str:
    by_strategy = {
        row["strategy"]: row
        for row in rows(ENDPOINTS)
        if row.get("target_ratio") == "0.258" and row.get("artifact_file_bytes")
    }
    missing = [strategy for strategy, _ in ENDPOINT_ORDER if strategy not in by_strategy]
    if missing:
        raise SystemExit(f"missing endpoint rows: {missing}")
    output = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Endpoint & Natural B & Final B & $\Delta$NLL & $\Delta$PPL & norm. $H$ \\",
        r"\midrule",
    ]
    for strategy, label in ENDPOINT_ORDER:
        row = by_strategy[strategy]
        natural_text = f"{int(float(row['artifact_natural_file_bytes'])):,}"
        byte_text = f"{int(float(row['artifact_file_bytes'])):,}"
        nll = float(row["nll_delta"])
        ppl = float(row["perplexity_delta"])
        hessian = float(row["normalized_hessian_cost"])
        if strategy == "Q+L":
            natural_text = rf"\textbf{{{natural_text}}}"
            byte_text = rf"\textbf{{{byte_text}}}"
            nll_text = rf"\textbf{{{nll:+.4f}}}"
            ppl_text = rf"\textbf{{{ppl:+.3f}}}"
            hessian_text = rf"\textbf{{{hessian:.5f}}}"
        else:
            nll_text = f"{nll:+.4f}"
            ppl_text = f"{ppl:+.3f}"
            hessian_text = f"{hessian:.5f}"
        output.append(
            f"{label} & {natural_text} & {byte_text} & {nll_text} & {ppl_text} & {hessian_text} \\\\"
        )
    output.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(output)


def repair_table() -> str:
    by_id = {row["comparison_id"]: row for row in rows(PAIRS)}
    missing = [key for key, _ in PAIR_ORDER if key not in by_id]
    if missing:
        raise SystemExit(f"missing paired comparison rows: {missing}")
    output = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Comparison & $\Delta$ bytes & NLL recovery & 95\% interval & Wins \\",
        r"\midrule",
    ]
    for key, label in PAIR_ORDER:
        row = by_id[key]
        byte_delta = int(row["artifact_file_byte_difference"])
        recovery = -float(row["paired_mean_nll_difference"])
        low = -float(row["normal_95_ci_high"])
        high = -float(row["normal_95_ci_low"])
        wins = f"{row['left_wins_lower_nll']}/{row['window_count']}"
        output.append(
            f"{label} & {byte_delta:+,} & {recovery:+.4f} & "
            f"[{low:+.4f}, {high:+.4f}] & {wins} \\\\"
        )
    output.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(output)


def verified_numbers() -> str:
    endpoints = {
        row["strategy"]: row
        for row in rows(ENDPOINTS)
        if row.get("target_ratio") == "0.258" and row.get("artifact_file_bytes")
    }
    comparisons = {row["comparison_id"]: row for row in rows(PAIRS)}
    comfort = rows(COMFORT)
    run_config = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    strict = endpoints["Q+S+L_QL_budget_component_scale"]
    relaxed = endpoints["Q+S+L_component_scale"]
    ql = endpoints["Q+L"]
    obs = comparisons["obs_vs_qs"]
    block = comparisons["block_scale_vs_q"]
    strict_pair = comparisons["constrained_scaled_qsl_vs_ql"]
    relaxed_pair = comparisons["unconstrained_scaled_qsl_vs_ql"]
    commands = {
        "VerifiedModel": "Pythia-70M",
        "VerifiedTensorCount": str(len(run_config["selected_layers"])),
        "VerifiedEvalTokens": f"{int(run_config['actual_eval_tokens']):,}",
        "DensePPL": f"{float(run_config['baseline_metrics']['perplexity']):.4f}",
        "QLBytes": f"{int(float(ql['artifact_file_bytes'])):,}",
        "QLNaturalBytes": f"{int(float(ql['artifact_natural_file_bytes'])):,}",
        "QLDeltaPPL": f"{float(ql['perplexity_delta']):.3f}",
        "StrictQSLNaturalBytes": f"{int(float(strict['artifact_natural_file_bytes'])):,}",
        "StrictQSLTailPaddingBytes": f"{int(float(strict['artifact_tail_padding_bytes'])):,}",
        "StrictQSLDeltaPPL": f"{float(strict['perplexity_delta']):.3f}",
        "StrictQSLMinusQLPPL": f"{float(strict_pair['endpoint_perplexity_difference']):+.3f}",
        "RelaxedQSLExtraBytes": f"{int(relaxed_pair['artifact_file_byte_difference']):,}",
        "RelaxedQSLDeltaPPL": f"{float(relaxed['perplexity_delta']):.3f}",
        "RelaxedQSLMinusQLPPL": f"{float(relaxed_pair['endpoint_perplexity_difference']):+.3f}",
        "StrictSLRho": f"{float(strict['rho_sl']):+.3f}",
        "StrictQSRho": f"{float(strict['rho_qs']):+.3f}",
        "StrictQLRho": f"{float(strict['rho_ql']):+.3f}",
        "ProxyCorrMin": f"{min(float(row['hessian_proxy_nll_correlation']) for row in comfort):.4f}",
        "ProxyCorrMax": f"{max(float(row['hessian_proxy_nll_correlation']) for row in comfort):.4f}",
        "OBSRecovery": f"{-float(obs['paired_mean_nll_difference']):.4f}",
        "OBSWins": f"{obs['left_wins_lower_nll']}/{obs['window_count']}",
        "BlockRecovery": f"{-float(block['paired_mean_nll_difference']):.4f}",
        "BlockExtraBytes": f"{int(block['artifact_file_byte_difference']):,}",
        "StrictQSLWins": f"{strict_pair['left_wins_lower_nll']}/{strict_pair['window_count']}",
    }
    return "\n".join(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in commands.items()
    )


def main() -> None:
    endpoint_path = OUT / "verified_exact_rate_table.tex"
    repair_path = OUT / "verified_repair_table.tex"
    numbers_path = OUT / "verified_numbers.tex"
    write_text(endpoint_path, endpoint_table())
    write_text(repair_path, repair_table())
    write_text(numbers_path, verified_numbers())
    manifest = {
        "schema_version": 1,
        "evidence_status": "verified_local_experiment",
        "scope": "Pythia-70M; six selected MLP linear tensors; one seed",
        "sources": {
            str(ENDPOINTS.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                ENDPOINTS.read_bytes()
            ).hexdigest(),
            str(PAIRS.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                PAIRS.read_bytes()
            ).hexdigest(),
            str(COMFORT.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                COMFORT.read_bytes()
            ).hexdigest(),
            str(RUN_CONFIG.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                RUN_CONFIG.read_bytes()
            ).hexdigest(),
        },
        "outputs": [endpoint_path.name, repair_path.name, numbers_path.name],
    }
    write_text(OUT / "verified_tables_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    print(endpoint_path)
    print(repair_path)
    print(numbers_path)


if __name__ == "__main__":
    main()
