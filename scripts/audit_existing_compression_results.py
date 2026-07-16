from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_spectral_dynamics.structured.hessian_repair import exact_payload_accounting


CORE_STRATEGIES = {
    "residual_q_only": "Q",
    "residual_q_l_same_budget": "Q+L",
    "residual_q_s_same_budget": "Q+S",
    "residual_q_s_l_same_budget": "Q+S+L",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def weighted_mean(rows: list[dict[str, str]], column: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        raw = row.get(column, "")
        if raw in {"", None}:
            continue
        weight = max(as_float(row.get("layer_parameter_count"), 1.0), 1.0)
        numerator += weight * as_float(raw)
        denominator += weight
    return numerator / denominator if denominator else float("nan")


def infer_shape(row: dict[str, str], input_dims: dict[str, int]) -> tuple[int, int]:
    name = str(row["layer"])
    cols = int(input_dims[name])
    parameters = as_int(row["layer_parameter_count"])
    if parameters <= 0 or parameters % cols:
        raise ValueError(f"cannot infer shape for {name}: parameters={parameters}, input_dim={cols}")
    return parameters // cols, cols


def payload_for_selection(
    selection: list[dict[str, str]],
    input_dims: dict[str, int],
    *,
    support_encoding: str,
) -> dict[str, int | float | str]:
    reference_bits = 0
    total_bits = 0
    component_bits: defaultdict[str, int] = defaultdict(int)
    encodings: set[str] = set()
    for row in selection:
        shape = infer_shape(row, input_dims)
        rows, _ = shape
        parameters = rows * shape[1]
        bits = as_int(row.get("bits"), 4)
        nonzero = as_int(as_float(row.get("sparse_keep_fraction")) * parameters)
        rank = as_int(row.get("lowrank_rank"), 0)
        payload = exact_payload_accounting(
            shape,
            base_code_bits=bits,
            base_scale_count=rows,
            base_scale_bits=16,
            sparse_nonzero=nonzero,
            sparse_value_bits=16,
            support_encoding=support_encoding,
            lowrank_rank=rank,
            lowrank_factor_bits=16,
        )
        fields = payload.as_dict()
        reference_bits += payload.reference_bits
        total_bits += payload.total_bits
        encodings.add(payload.support_encoding)
        for key, value in fields.items():
            if key.endswith("_stored_bits"):
                component_bits[key] += int(value)
    ratio = total_bits / max(reference_bits, 1)
    return {
        "reference_bits": reference_bits,
        "payload_bits": total_bits,
        "payload_ratio": ratio,
        "compression_ratio": reference_bits / max(total_bits, 1),
        "support_encodings": "+".join(sorted(encodings)),
        **component_bits,
    }


def audit_qwen_run(run_dir: Path, target_ratio: float, tolerance: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics = run_dir / "metrics"
    strategy_rows = read_csv(metrics / "residual_stack_strategy.csv")
    selection_rows = read_csv(metrics / "residual_stack_selection.csv")
    covariance_rows = read_csv(metrics / "activation_covariance_summary.csv")
    input_dims = {row["layer"]: as_int(row["input_dim"]) for row in covariance_rows}
    families = {row.get("layer_family", "") for row in selection_rows}
    run_label = "Qwen2-7B attn+MLP" if "mlp" in families else "Qwen2-7B attention"
    strategy_lookup = {row["strategy"]: row for row in strategy_rows}
    audit_rows: list[dict[str, object]] = []
    rho_rows: list[dict[str, object]] = []
    for strategy, label in CORE_STRATEGIES.items():
        selected = [row for row in selection_rows if row.get("strategy") == strategy]
        endpoint = strategy_lookup.get(strategy)
        if not selected or endpoint is None:
            continue
        lower = payload_for_selection(selected, input_dims, support_encoding="entropy")
        csr = payload_for_selection(selected, input_dims, support_encoding="csr_fixed")
        nominal = as_float(endpoint.get("nominal_memory_ratio"))
        entropy_ratio = float(lower["payload_ratio"])
        csr_ratio = float(csr["payload_ratio"])
        audit_rows.append(
            {
                "run": run_label,
                "result_dir": run_dir.name,
                "strategy": strategy,
                "method": label,
                "target_ratio": target_ratio,
                "nominal_ratio": nominal,
                "entropy_lower_bound_ratio": entropy_ratio,
                "csr16_ratio": csr_ratio,
                "nominal_underestimate_entropy": entropy_ratio - nominal,
                "nominal_underestimate_csr16": csr_ratio - nominal,
                "strict_target_match_entropy": abs(entropy_ratio - target_ratio) / target_ratio <= tolerance,
                "strict_target_match_csr16": abs(csr_ratio - target_ratio) / target_ratio <= tolerance,
                "nll": as_float(endpoint.get("nll")),
                "ppl": as_float(endpoint.get("perplexity")),
                "ppl_delta": as_float(endpoint.get("signed_ppl_delta")),
                "tokens": as_int(endpoint.get("tokens")),
                "predicted_hessian_cost": as_float(endpoint.get("predicted_hessian_cost")),
                "q_scale_bits": int(lower.get("base_scales_stored_bits", 0)),
                "sparse_value_bits": int(lower.get("sparse_values_stored_bits", 0)),
                "sparse_entropy_support_bits": int(lower.get("sparse_support_stored_bits", 0)),
                "sparse_csr_support_bits": int(csr.get("sparse_support_stored_bits", 0)),
                "lowrank_factor_bits": int(lower.get("lowrank_factors_stored_bits", 0)),
                "entropy_support_encodings": lower["support_encodings"],
                "csr_support_encodings": csr["support_encodings"],
            }
        )
        for column, pair in (
            ("rho_q_error_s_res", "Qerr,Sres"),
            ("rho_q_error_l_res", "Qerr,Lres"),
            ("rho_q_plus_s_error_l_res", "Q+S err,Lres"),
            ("rho_s_res_l_res", "Sres,Lres"),
        ):
            value = weighted_mean(selected, column)
            if math.isfinite(value):
                rho_rows.append(
                    {
                        "run": run_label,
                        "method": label,
                        "pair": pair,
                        "rho_h": value,
                        "abs_rho_h": abs(value),
                        "interpretation": "orthogonal" if abs(value) <= 0.05 else ("repair_cancellation" if value < 0 else "conflict"),
                    }
                )
    by_method = {str(row["method"]): row for row in audit_rows}
    q_row = by_method.get("Q")
    if q_row is not None:
        for row in audit_rows:
            extra = float(row["entropy_lower_bound_ratio"]) - float(q_row["entropy_lower_bound_ratio"])
            gain = float(q_row["ppl_delta"]) - float(row["ppl_delta"])
            row["ppl_gain_vs_q"] = gain
            row["ppl_gain_per_1pct_dense_entropy"] = gain / (100.0 * extra) if extra > 0.0 else float("nan")
    return audit_rows, rho_rows


def stability_audit(aggregate_path: Path) -> list[dict[str, object]]:
    rows = read_csv(aggregate_path)
    output: list[dict[str, object]] = []
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("strategy") in CORE_STRATEGIES:
            grouped[row["run"]].append(row)
    for run, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda row: as_float(row["ppl_delta"]))
        ranks = {row["strategy"]: index + 1 for index, row in enumerate(ordered)}
        best_single = min(
            (row for row in candidates if row["strategy"] != "residual_q_s_l_same_budget"),
            key=lambda row: as_float(row["ppl_delta"]),
        )
        qsl = next(row for row in candidates if row["strategy"] == "residual_q_s_l_same_budget")
        for row in candidates:
            output.append(
                {
                    "run": run,
                    "method": CORE_STRATEGIES[row["strategy"]],
                    "strategy": row["strategy"],
                    "nominal_ratio": as_float(row["memory_ratio"]),
                    "ppl": as_float(row["perplexity"]),
                    "ppl_delta": as_float(row["ppl_delta"]),
                    "rank_within_core": ranks[row["strategy"]],
                    "qsl_gain_vs_best_single": as_float(best_single["ppl_delta"]) - as_float(qsl["ppl_delta"]),
                    "best_single_method": CORE_STRATEGIES[best_single["strategy"]],
                }
            )
    return output


def rate_matched_pairs(payload_rows: list[dict[str, object]], tolerance: float) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in payload_rows:
        grouped[str(row["run"])].append(row)
    for run, rows in grouped.items():
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                for codec_column, codec in (
                    ("entropy_lower_bound_ratio", "entropy_lower_bound"),
                    ("csr16_ratio", "csr16"),
                ):
                    left_rate = float(left[codec_column])
                    right_rate = float(right[codec_column])
                    gap = abs(left_rate - right_rate) / max(left_rate, right_rate, 1e-12)
                    output.append(
                        {
                            "run": run,
                            "codec": codec,
                            "method_a": left["method"],
                            "method_b": right["method"],
                            "rate_a": left_rate,
                            "rate_b": right_rate,
                            "relative_rate_gap": gap,
                            "pairwise_rate_match": gap <= tolerance,
                            "ppl_delta_a": left["ppl_delta"],
                            "ppl_delta_b": right["ppl_delta"],
                            "winner": left["method"] if float(left["ppl_delta"]) < float(right["ppl_delta"]) else right["method"],
                        }
                    )
    return output


def evidence_flags(repo_root: Path, payload_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    qwen_dirs = sorted((repo_root / "results").glob("residual_stack_validate_Qwen_Qwen2-7B_*"))
    for run_dir in qwen_dirs:
        config = json.loads((run_dir / "metrics" / "run_config.json").read_text(encoding="utf-8"))
        runtime = config.get("runtime_args", {})
        strategy = read_csv(run_dir / "metrics" / "residual_stack_strategy.csv")
        baseline = next(row for row in strategy if row["strategy"] == "baseline")
        run_label = next(
            str(row["run"]) for row in payload_rows if str(row["result_dir"]) == run_dir.name
        )
        checks = [
            ("single_seed", True, f"seed={runtime.get('seed', config.get('seed', 17))}; no repeated run is committed"),
            (
                "shared_calibration_and_eval",
                not bool(runtime.get("disjoint_text_splits", False)),
                f"text_split_policy={config.get('text_split_policy', 'shared_text_pool')}",
            ),
            ("low_token_count", as_int(baseline.get("tokens")) < 1024, f"tokens={baseline.get('tokens')}"),
            ("no_per_sample_nll", True, "only aggregate nll/ppl is stored; bootstrap CI is impossible"),
        ]
        for flag, active, evidence in checks:
            flags.append({"run": run_label, "flag": flag, "active": active, "evidence": evidence})
    aggregate = read_csv(repo_root / "results" / "compare_7b_dam_residual_stack_20260707" / "strategy_comparison.csv")
    for run in ("Pythia-70M", "Pythia-160M"):
        result_dirs = {row["result_dir"] for row in aggregate if row["run"] == run}
        missing = [name for name in result_dirs if not (repo_root / "results" / name).exists()]
        flags.append(
            {
                "run": run,
                "flag": "raw_run_missing",
                "active": bool(missing),
                "evidence": ",".join(missing) if missing else "",
            }
        )
    return flags


def render_summary(
    payload_rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
    rho_rows: list[dict[str, object]],
) -> tuple[dict[str, object], str]:
    by_run_method = {(str(row["run"]), str(row["method"])): row for row in payload_rows}
    qwen_runs = sorted({str(row["run"]) for row in payload_rows})
    qsl_wins = sum(
        1
        for row in stability_rows
        if row["method"] == "Q+S+L" and as_int(row["rank_within_core"]) == 1
    )
    qsl_runs = sum(1 for row in stability_rows if row["method"] == "Q+S+L")
    q_s_rhos = [float(row["rho_h"]) for row in rho_rows if row["method"] in {"Q+S", "Q+S+L"} and row["pair"] == "Qerr,Sres"]
    summary: dict[str, object] = {
        "scope": "Exploratory audit of committed results; no new endpoint is implied.",
        "qsl_core_wins": qsl_wins,
        "qsl_core_runs": qsl_runs,
        "mean_q_s_hessian_rho": sum(q_s_rhos) / max(len(q_s_rhos), 1),
        "strict_rate_tolerance": 0.01,
        "conclusions": [
            "The current Q+S+L advantage is run-dependent rather than universal.",
            "Negative Q/S Hessian cosine is repair cancellation, not orthogonality.",
            "Sparse index/value overhead invalidates the nominal 0.258 matched-rate claim.",
            "Q+L is closest to its nominal payload and is the most credible current parameter-efficient repair.",
        ],
    }
    lines = [
        "# Existing-result exact-payload audit",
        "",
        "This artifact audits already committed endpoints. It does not turn the old nominal-rate experiments into new exact-rate evidence.",
        "",
        "| run | method | nominal | entropy lower bound | CSR16 | PPL delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload_rows:
        lines.append(
            f"| {row['run']} | {row['method']} | {float(row['nominal_ratio']):.6f} | "
            f"{float(row['entropy_lower_bound_ratio']):.6f} | {float(row['csr16_ratio']):.6f} | "
            f"{float(row['ppl_delta']):+.6f} |"
        )
    lines += [
        "",
        f"Q+S+L ranks first within Q/Q+L/Q+S/Q+S+L in {qsl_wins}/{qsl_runs} committed runs, but the winning margin is not stable.",
        "",
        "The Q/S interaction is predominantly negative. That is useful error cancellation, but it must not be reported as Hessian orthogonality.",
        "",
        "Sparse methods exceed the nominal rate after storing FP16 residual values and support. Exact-rate reruns must reduce nnz and re-evaluate NLL/PPL.",
    ]
    if qwen_runs:
        attention = by_run_method.get(("Qwen2-7B attention", "Q+L"))
        mlp_qs = by_run_method.get(("Qwen2-7B attn+MLP", "Q+S"))
        mlp_qsl = by_run_method.get(("Qwen2-7B attn+MLP", "Q+S+L"))
        if attention:
            lines.append(
                f"Qwen attention-only currently favors Q+L (PPL delta {float(attention['ppl_delta']):+.6f}); "
                "its payload correction is small because no sparse support is stored."
            )
        if mlp_qs and mlp_qsl:
            lines.append(
                f"Qwen attn+MLP Q+S+L beats Q+S by only "
                f"{float(mlp_qs['ppl_delta']) - float(mlp_qsl['ppl_delta']):.6f} PPL before exact-rate rerunning."
            )
    return summary, "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit committed compression results with exact payload accounting.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "exact_rate_hessian_repair_20260713",
    )
    parser.add_argument("--target-ratio", type=float, default=0.258)
    parser.add_argument("--rate-tolerance", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    payload_rows: list[dict[str, object]] = []
    rho_rows: list[dict[str, object]] = []
    for run_dir in sorted((repo_root / "results").glob("residual_stack_validate_Qwen_Qwen2-7B_*")):
        current_payload, current_rho = audit_qwen_run(run_dir, args.target_ratio, args.rate_tolerance)
        payload_rows.extend(current_payload)
        rho_rows.extend(current_rho)
    stability_rows = stability_audit(
        repo_root / "results" / "compare_7b_dam_residual_stack_20260707" / "strategy_comparison.csv"
    )
    pair_rows = rate_matched_pairs(payload_rows, args.rate_tolerance)
    flags = evidence_flags(repo_root, payload_rows)
    summary, markdown = render_summary(payload_rows, stability_rows, rho_rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "payload_audit.csv", payload_rows)
    write_csv(output / "strategy_stability.csv", stability_rows)
    write_csv(output / "hessian_interactions.csv", rho_rows)
    write_csv(output / "rate_matched_pairs.csv", pair_rows)
    write_csv(output / "evidence_flags.csv", flags)
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "audit_summary.md").write_text(markdown, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
