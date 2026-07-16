#!/usr/bin/env python3
"""Summarize paired endpoint evidence for the serialized Hessian probe.

The script intentionally uses only the Python standard library.  It joins the
fixed-window NLL records to the evaluated endpoint rows and the independently
decoded artifact manifest, validates the physical-byte claims, and emits two
deterministic machine-readable summaries.

All method differences use ``left - right``.  Lower NLL/perplexity is better,
so a negative difference favors the left-hand method.  The normal intervals
and exact sign tests are descriptive diagnostics over the 16 fixed contiguous
windows; they are not causal or population-level inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "serialized_hessian_paired_summary.v1"
DEFAULT_RESULT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "pretrained_hessian_repair_pythia70m_serialized_20260714"
)
NORMAL_95_Z = 1.96
SIGN_TEST_ALPHA = 0.05


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    left: str
    right: str
    purpose: str


COMPARISONS = (
    ComparisonSpec(
        "constrained_scaled_qsl_vs_ql",
        "Q+S+L_QL_budget_component_scale",
        "Q+L",
        "strict equal-serialized-byte Q/S/L combination with folded component scaling",
    ),
    ComparisonSpec(
        "constrained_unscaled_qsl_vs_ql",
        "Q+S+L_QL_budget",
        "Q+L",
        "strict equal-serialized-byte Q/S/L combination without component scaling",
    ),
    ComparisonSpec(
        "unconstrained_scaled_qsl_vs_ql",
        "Q+S+L_component_scale",
        "Q+L",
        "scaled Q/S/L endpoint without the Q+L serialized-byte cap",
    ),
    ComparisonSpec(
        "obs_vs_qs",
        "Q+S_OBS",
        "Q+S",
        "OBS value repair on the same frozen sparse support and serialized payload",
    ),
    ComparisonSpec(
        "block_scale_vs_q",
        "Q_block_scale",
        "Q",
        "block-scale protection versus row-scale quantization",
    ),
)


CSV_FIELDS = (
    "comparison_id",
    "purpose",
    "left_strategy",
    "right_strategy",
    "difference_convention",
    "window_count",
    "paired_mean_nll_difference",
    "paired_nll_difference_se",
    "normal_95_ci_low",
    "normal_95_ci_high",
    "left_wins_lower_nll",
    "right_wins_lower_nll",
    "ties",
    "exact_two_sided_sign_p",
    "normal_ci_excludes_zero",
    "sign_test_p_below_0_05",
    "uncertainty_label",
    "descriptive_direction",
    "left_endpoint_nll",
    "right_endpoint_nll",
    "endpoint_nll_difference",
    "left_endpoint_perplexity",
    "right_endpoint_perplexity",
    "endpoint_perplexity_difference",
    "left_artifact_file_bytes",
    "right_artifact_file_bytes",
    "artifact_file_byte_difference",
    "left_artifact_natural_file_bytes",
    "right_artifact_natural_file_bytes",
    "artifact_natural_file_byte_difference",
    "serialized_budget_relation",
    "interpretation",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_int(value: str, *, field: str) -> int:
    try:
        parsed = Decimal(value)
    except Exception as exc:  # pragma: no cover - Decimal supplies useful detail
        raise ValueError(f"invalid integer field {field}: {value!r}") from exc
    integral = parsed.to_integral_value()
    if parsed != integral:
        raise ValueError(f"non-integral field {field}: {value!r}")
    return int(integral)


def exact_two_sided_sign_test(differences: Iterable[float]) -> dict[str, int | float]:
    """Return the exact two-sided p-value for non-tied signs.

    Ties are excluded from the Binomial(n, 0.5) null, while their count remains
    explicit in the result.  No continuity correction or asymptotic
    approximation is used.
    """

    values = [float(value) for value in differences]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("sign test received a non-finite difference")
    left_wins = sum(value < 0.0 for value in values)
    right_wins = sum(value > 0.0 for value in values)
    ties = len(values) - left_wins - right_wins
    non_ties = left_wins + right_wins
    if non_ties == 0:
        p_value = 1.0
    else:
        tail = min(left_wins, right_wins)
        tail_probability = sum(math.comb(non_ties, count) for count in range(tail + 1)) / (2**non_ties)
        p_value = min(1.0, 2.0 * tail_probability)
    return {
        "left_wins_lower_nll": left_wins,
        "right_wins_lower_nll": right_wins,
        "ties": ties,
        "non_tied_window_count": non_ties,
        "exact_two_sided_sign_p": p_value,
    }


def _endpoint_rows(result_dir: Path) -> dict[str, dict[str, str]]:
    evaluated = [
        row
        for row in _read_csv(result_dir / "strategy_endpoints.csv")
        if row.get("heldout_evaluated", "").strip().lower() == "true"
    ]
    by_strategy = {row["strategy"]: row for row in evaluated}
    if len(by_strategy) != len(evaluated):
        raise ValueError("duplicate held-out endpoint strategy")
    required = {spec.left for spec in COMPARISONS} | {spec.right for spec in COMPARISONS}
    missing = sorted(required - set(by_strategy))
    if missing:
        raise ValueError(f"missing evaluated endpoints: {missing}")
    target_ratios = {float(row["target_ratio"]) for row in evaluated}
    if len(target_ratios) != 1:
        raise ValueError(f"evaluated endpoints do not share one target ratio: {target_ratios}")
    return by_strategy


def _window_rows(result_dir: Path) -> dict[str, dict[tuple[int, int, int], dict[str, str]]]:
    grouped: dict[str, dict[tuple[int, int, int], dict[str, str]]] = defaultdict(dict)
    for row in _read_csv(result_dir / "endpoint_window_nll.csv"):
        key = (
            _exact_int(row["sequence_index"], field="sequence_index"),
            _exact_int(row["window_index"], field="window_index"),
            _exact_int(row["batch_index"], field="batch_index"),
        )
        strategy_rows = grouped[row["strategy"]]
        if key in strategy_rows:
            raise ValueError(f"duplicate window key for {row['strategy']}: {key}")
        strategy_rows[key] = row
    return dict(grouped)


def _artifact_rows(result_dir: Path, endpoints: Mapping[str, Mapping[str, str]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = result_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strategies = manifest.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("artifact manifest has no strategy entries")
    by_strategy = {str(row["strategy"]): row for row in strategies}
    if len(by_strategy) != len(strategies):
        raise ValueError("duplicate artifact strategy")

    for strategy, endpoint in endpoints.items():
        if strategy not in by_strategy:
            raise ValueError(f"manifest missing evaluated strategy: {strategy}")
        artifact = by_strategy[strategy]
        for field in ("artifact_file_bytes", "artifact_natural_file_bytes"):
            endpoint_value = _exact_int(endpoint[field], field=f"endpoint.{strategy}.{field}")
            manifest_value = int(artifact[field])
            if endpoint_value != manifest_value:
                raise ValueError(
                    f"endpoint/manifest mismatch for {strategy}.{field}: "
                    f"{endpoint_value} != {manifest_value}"
                )
        for field in ("artifact_path", "artifact_sha256"):
            if endpoint[field] != str(artifact[field]):
                raise ValueError(f"endpoint/manifest mismatch for {strategy}.{field}")
        artifact_path = result_dir / str(artifact["artifact_path"])
        actual_size = artifact_path.stat().st_size
        if actual_size != int(artifact["artifact_file_bytes"]):
            raise ValueError(
                f"artifact size mismatch for {strategy}: {actual_size} != "
                f"{artifact['artifact_file_bytes']}"
            )
        if _sha256(artifact_path) != str(artifact["artifact_sha256"]):
            raise ValueError(f"artifact SHA256 mismatch for {strategy}")

    reference = manifest.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("artifact manifest has no reference entry")
    reference_path = result_dir / str(reference["path"])
    actual_reference_size = reference_path.stat().st_size
    if actual_reference_size != int(reference["file_bytes"]):
        raise ValueError(
            f"reference artifact size mismatch: {actual_reference_size} != {reference['file_bytes']}"
        )
    if _sha256(reference_path) != str(reference["sha256"]):
        raise ValueError("reference artifact SHA256 mismatch")
    return by_strategy, manifest


def _validate_physical_rate_claims(
    endpoints: Mapping[str, Mapping[str, str]],
    artifacts: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    ql_name = "Q+L"
    constrained = (
        "Q+S+L_QL_budget",
        "Q+S+L_QL_budget_component_scale",
    )
    unconstrained = "Q+S+L_component_scale"
    ql_bytes = int(artifacts[ql_name]["artifact_file_bytes"])
    ql_natural_bytes = int(artifacts[ql_name]["artifact_natural_file_bytes"])
    if ql_bytes != ql_natural_bytes:
        raise ValueError("Q+L unexpectedly contains tail padding")
    if not bool(artifacts[ql_name]["same_physical_bytes_as_ql"]):
        raise ValueError("Q+L is not marked as its own physical-byte reference")
    for strategy, endpoint in endpoints.items():
        if _exact_int(endpoint["ql_budget_file_bytes"], field=f"{strategy}.ql_budget_file_bytes") != ql_bytes:
            raise ValueError(f"endpoint {strategy} has a different Q+L byte budget")

    constrained_rows: dict[str, Any] = {}
    for strategy in constrained:
        artifact = artifacts[strategy]
        file_bytes = int(artifact["artifact_file_bytes"])
        natural_bytes = int(artifact["artifact_natural_file_bytes"])
        if file_bytes != ql_bytes:
            raise ValueError(f"strict control {strategy} is not byte-equal to Q+L")
        if natural_bytes > ql_bytes:
            raise ValueError(f"strict control {strategy} exceeds Q+L before padding")
        if not bool(artifact["same_physical_bytes_as_ql"]):
            raise ValueError(f"strict control {strategy} is not marked byte-equal")
        if endpoints[strategy]["same_physical_bytes_as_ql"].strip().lower() != "true":
            raise ValueError(f"endpoint {strategy} is not marked byte-equal")
        constrained_rows[strategy] = {
            "artifact_file_bytes": file_bytes,
            "artifact_natural_file_bytes": natural_bytes,
            "tail_padding_bytes": int(artifact["artifact_tail_padding_bytes"]),
            "file_byte_difference_vs_ql": file_bytes - ql_bytes,
            "natural_file_byte_difference_vs_ql": natural_bytes - ql_bytes,
        }

    overage = int(artifacts[unconstrained]["artifact_file_bytes"]) - ql_bytes
    if overage <= 0:
        raise ValueError("unconstrained scaled Q/S/L does not exceed the Q+L byte budget")
    if bool(artifacts[unconstrained]["same_physical_bytes_as_ql"]):
        raise ValueError("unconstrained scaled Q/S/L is incorrectly marked byte-equal")
    if endpoints[unconstrained]["same_physical_bytes_as_ql"].strip().lower() != "false":
        raise ValueError("unconstrained endpoint is incorrectly marked byte-equal")

    return {
        "artifact_scope": manifest.get("scope"),
        "production_backend": bool(manifest.get("production_backend")),
        "alignment_bytes": int(manifest["alignment_bytes"]),
        "serialized_rate_cap_enforced": bool(manifest["serialized_rate_cap_enforced"]),
        "endpoint_manifest_fields_consistent": True,
        "artifact_file_sizes_match_manifest": True,
        "artifact_sha256_digests_match_manifest": True,
        "reference_artifact_file_bytes": int(manifest["reference"]["file_bytes"]),
        "ql_reference": {
            "strategy": ql_name,
            "artifact_file_bytes": ql_bytes,
            "artifact_natural_file_bytes": ql_natural_bytes,
        },
        "strict_equal_byte_controls": constrained_rows,
        "unconstrained_scaled_qsl": {
            "strategy": unconstrained,
            "artifact_file_bytes": int(artifacts[unconstrained]["artifact_file_bytes"]),
            "artifact_natural_file_bytes": int(artifacts[unconstrained]["artifact_natural_file_bytes"]),
            "file_byte_overage_vs_ql": overage,
            "file_byte_overage_fraction_vs_ql": overage / ql_bytes,
        },
    }


def _budget_relation(byte_difference: int) -> str:
    if byte_difference == 0:
        return "equal_serialized_bytes"
    if byte_difference > 0:
        return "left_uses_more_serialized_bytes"
    return "left_uses_fewer_serialized_bytes"


def _validate_execution_evidence(result_dir: Path) -> dict[str, Any]:
    files = {
        "status": result_dir / "formal_run.status",
        "exit_code": result_dir / "formal_run.exit_code",
        "time": result_dir / "formal_run.time.txt",
        "stdout": result_dir / "formal_run.stdout.log",
        "stderr": result_dir / "formal_run.stderr.log",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing formal run evidence: {missing}")
    status = files["status"].read_text(encoding="utf-8").strip()
    exit_code = int(files["exit_code"].read_text(encoding="utf-8").strip())
    timing = files["time"].read_text(encoding="utf-8").strip()
    timing_match = re.fullmatch(r"elapsed=([0-9.]+) maxrss_kb=([0-9]+) exit=([0-9]+)", timing)
    if status != "COMPLETED" or exit_code != 0 or timing_match is None:
        raise ValueError(
            f"formal run did not pass completion gate: status={status!r}, "
            f"exit_code={exit_code}, timing={timing!r}"
        )
    timing_exit = int(timing_match.group(3))
    if timing_exit != exit_code:
        raise ValueError("formal run timing/exit_code files disagree")
    expected_stdout = "results/pretrained_hessian_repair_pythia70m_serialized_20260714"
    stdout = files["stdout"].read_text(encoding="utf-8").strip()
    if stdout != expected_stdout:
        raise ValueError(f"unexpected formal run stdout: {stdout!r}")
    return {
        "status": status,
        "exit_code": exit_code,
        "elapsed_seconds": float(timing_match.group(1)),
        "maxrss_kb": int(timing_match.group(2)),
        "stdout_result_path": stdout,
        "files": {
            name: {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in sorted(files.items())
        },
    }


def _interpretation(
    *,
    spec: ComparisonSpec,
    mean: float,
    ci_low: float,
    ci_high: float,
    sign_p: float,
    byte_difference: int,
    window_count: int,
) -> tuple[str, str, str]:
    if mean < 0.0:
        direction = "left_lower_nll"
        direction_text = f"{spec.left} has lower mean NLL than {spec.right}"
    elif mean > 0.0:
        direction = "left_higher_nll"
        direction_text = f"{spec.left} has higher mean NLL than {spec.right}"
    else:
        direction = "equal_mean_nll"
        direction_text = f"{spec.left} and {spec.right} have equal mean NLL"

    if byte_difference == 0:
        budget_text = "at equal serialized bytes"
    elif byte_difference > 0:
        budget_text = f"while using {byte_difference} more serialized bytes"
    else:
        budget_text = f"while using {-byte_difference} fewer serialized bytes"

    ci_inconclusive = ci_low <= 0.0 <= ci_high
    sign_inconclusive = sign_p >= SIGN_TEST_ALPHA
    uncertainty_label = (
        "inconclusive"
        if ci_inconclusive or sign_inconclusive
        else "fixed_window_diagnostics_agree"
    )
    if ci_inconclusive and sign_inconclusive:
        diagnostic_text = "the normal interval includes zero and the exact sign test is inconclusive"
    elif ci_inconclusive:
        diagnostic_text = "the normal interval includes zero, so uncertainty remains despite the sign test"
    elif sign_inconclusive:
        diagnostic_text = "the normal interval excludes zero, but the exact sign test is inconclusive"
    else:
        diagnostic_text = "the normal interval and exact sign test agree on these fixed windows"

    sentence = (
        f"Descriptive fixed-window comparison: {direction_text} {budget_text}; "
        f"{diagnostic_text}. This is non-causal evidence from {window_count} fixed contiguous "
        "windows, not a population-level confidence claim; no multiple-comparison correction "
        "was applied."
    )
    return direction, uncertainty_label, sentence


def _comparison_row(
    spec: ComparisonSpec,
    *,
    endpoints: Mapping[str, Mapping[str, str]],
    windows: Mapping[str, Mapping[tuple[int, int, int], Mapping[str, str]]],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if spec.left not in windows or spec.right not in windows:
        raise ValueError(f"window rows missing for {spec.comparison_id}")
    left_windows = windows[spec.left]
    right_windows = windows[spec.right]
    if set(left_windows) != set(right_windows):
        raise ValueError(f"paired window keys differ for {spec.comparison_id}")

    differences: list[float] = []
    for key in sorted(left_windows):
        left_row = left_windows[key]
        right_row = right_windows[key]
        left_tokens = _exact_int(left_row["tokens"], field=f"{spec.left}.tokens")
        right_tokens = _exact_int(right_row["tokens"], field=f"{spec.right}.tokens")
        if left_tokens != right_tokens:
            raise ValueError(f"paired token counts differ for {spec.comparison_id} at {key}")
        differences.append(float(left_row["nll"]) - float(right_row["nll"]))
    if len(differences) < 2:
        raise ValueError(f"at least two paired windows are required for {spec.comparison_id}")

    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(len(differences))
    ci_low = mean - NORMAL_95_Z * se
    ci_high = mean + NORMAL_95_Z * se
    sign = exact_two_sided_sign_test(differences)

    left_endpoint = endpoints[spec.left]
    right_endpoint = endpoints[spec.right]
    endpoint_nll_difference = float(left_endpoint["heldout_nll"]) - float(right_endpoint["heldout_nll"])
    if not math.isclose(endpoint_nll_difference, mean, rel_tol=0.0, abs_tol=2e-15):
        raise ValueError(
            f"window mean does not reproduce endpoint NLL difference for {spec.comparison_id}: "
            f"{mean} != {endpoint_nll_difference}"
        )
    left_artifact = artifacts[spec.left]
    right_artifact = artifacts[spec.right]
    file_byte_difference = int(left_artifact["artifact_file_bytes"]) - int(right_artifact["artifact_file_bytes"])
    natural_byte_difference = int(left_artifact["artifact_natural_file_bytes"]) - int(
        right_artifact["artifact_natural_file_bytes"]
    )
    direction, uncertainty_label, interpretation = _interpretation(
        spec=spec,
        mean=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        sign_p=float(sign["exact_two_sided_sign_p"]),
        byte_difference=file_byte_difference,
        window_count=len(differences),
    )

    return {
        "comparison_id": spec.comparison_id,
        "purpose": spec.purpose,
        "left_strategy": spec.left,
        "right_strategy": spec.right,
        "difference_convention": "left_minus_right; negative_nll_or_perplexity_favors_left",
        "window_count": len(differences),
        "paired_mean_nll_difference": mean,
        "paired_nll_difference_se": se,
        "normal_95_ci_low": ci_low,
        "normal_95_ci_high": ci_high,
        "left_wins_lower_nll": int(sign["left_wins_lower_nll"]),
        "right_wins_lower_nll": int(sign["right_wins_lower_nll"]),
        "ties": int(sign["ties"]),
        "exact_two_sided_sign_p": float(sign["exact_two_sided_sign_p"]),
        "normal_ci_excludes_zero": not (ci_low <= 0.0 <= ci_high),
        "sign_test_p_below_0_05": float(sign["exact_two_sided_sign_p"]) < SIGN_TEST_ALPHA,
        "uncertainty_label": uncertainty_label,
        "descriptive_direction": direction,
        "left_endpoint_nll": float(left_endpoint["heldout_nll"]),
        "right_endpoint_nll": float(right_endpoint["heldout_nll"]),
        "endpoint_nll_difference": endpoint_nll_difference,
        "left_endpoint_perplexity": float(left_endpoint["heldout_perplexity"]),
        "right_endpoint_perplexity": float(right_endpoint["heldout_perplexity"]),
        "endpoint_perplexity_difference": float(left_endpoint["heldout_perplexity"])
        - float(right_endpoint["heldout_perplexity"]),
        "left_artifact_file_bytes": int(left_artifact["artifact_file_bytes"]),
        "right_artifact_file_bytes": int(right_artifact["artifact_file_bytes"]),
        "artifact_file_byte_difference": file_byte_difference,
        "left_artifact_natural_file_bytes": int(left_artifact["artifact_natural_file_bytes"]),
        "right_artifact_natural_file_bytes": int(right_artifact["artifact_natural_file_bytes"]),
        "artifact_natural_file_byte_difference": natural_byte_difference,
        "serialized_budget_relation": _budget_relation(file_byte_difference),
        "interpretation": interpretation,
    }


def build_summary(result_dir: Path = DEFAULT_RESULT_DIR) -> dict[str, Any]:
    """Validate source evidence and return the deterministic summary object."""

    result_dir = Path(result_dir)
    endpoints = _endpoint_rows(result_dir)
    windows = _window_rows(result_dir)
    artifacts, manifest = _artifact_rows(result_dir, endpoints)
    physical_rate_validation = _validate_physical_rate_claims(endpoints, artifacts, manifest)
    execution_evidence = _validate_execution_evidence(result_dir)
    rows = [
        _comparison_row(
            spec,
            endpoints=endpoints,
            windows=windows,
            artifacts=artifacts,
        )
        for spec in COMPARISONS
    ]
    target_ratio = float(next(iter(endpoints.values()))["target_ratio"])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_from": [
            "strategy_endpoints.csv",
            "endpoint_window_nll.csv",
            "artifact_manifest.json",
            "artifacts/*.hrc (file-size and SHA256 validation)",
            "formal_run.{status,exit_code,time.txt,stdout.log,stderr.log}",
        ],
        "target_ratio": target_ratio,
        "difference_convention": "left_minus_right; negative_nll_or_perplexity_favors_left",
        "statistical_scope": {
            "normal_interval": "mean +/- 1.96 sample-standard-errors over paired fixed windows",
            "sign_test": "exact two-sided Binomial(n_non_ties, 0.5); ties excluded",
            "alpha": SIGN_TEST_ALPHA,
            "multiple_comparison_correction": "none",
            "inference_limit": (
                "Descriptive, non-causal diagnostics over fixed contiguous language-model windows; "
                "not an independence-based population confidence claim. A comparison is labelled "
                "inconclusive when either the normal interval includes zero or the exact sign-test "
                "p-value is at least 0.05."
            ),
        },
        "execution_evidence": execution_evidence,
        "physical_rate_validation": physical_rate_validation,
        "comparisons": rows,
    }


def render_json(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def render_csv(comparisons: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for comparison in comparisons:
        writer.writerow({field: comparison[field] for field in CSV_FIELDS})
    return buffer.getvalue()


def write_outputs(result_dir: Path = DEFAULT_RESULT_DIR, *, check: bool = False) -> tuple[Path, Path]:
    result_dir = Path(result_dir)
    summary = build_summary(result_dir)
    outputs = {
        result_dir / "serialized_rate_summary.json": render_json(summary),
        result_dir / "paired_method_comparisons.csv": render_csv(summary["comparisons"]),
    }
    for path, expected in outputs.items():
        if check:
            if not path.is_file():
                raise FileNotFoundError(f"missing generated output: {path}")
            actual = path.read_text(encoding="utf-8")
            if actual != expected:
                raise RuntimeError(f"generated output drift: {path}")
        else:
            path.write_text(expected, encoding="utf-8", newline="")
    return tuple(outputs)  # type: ignore[return-value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--check", action="store_true", help="fail if committed outputs differ")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = write_outputs(args.result_dir, check=args.check)
    action = "verified" if args.check else "wrote"
    for path in paths:
        print(f"{action}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
