from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from llm_spectral_dynamics.structured.codec_artifact import (
    ArtifactWriteResult,
    LayerCodecPayload,
    write_codec_artifact,
    write_fp16_reference_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_scaling_pilot",
    REPO_ROOT / "paper" / "results" / "build_scaling_pilot.py",
)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    assert rows
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _rho_kind(value: float) -> str:
    if math.isnan(value):
        return "inactive"
    if abs(value) <= 0.1:
        return "hessian_orthogonal"
    return "repair_cancellation" if value < 0 else "positive_conflict"


def _codec_layers(
    strategy: str, selected: list[str], model_index: int
) -> list[LayerCodecPayload]:
    """Build small, real HRC payloads with strategy-faithful component sets."""

    rows, columns = 16, 32
    has_sparse = "+S" in strategy
    has_lowrank = "+L" in strategy
    rank = 16 if strategy == "Q+L" else (1 if has_lowrank else 0)
    layers: list[LayerCodecPayload] = []
    for layer_index, name in enumerate(selected):
        code_seed = np.arange(rows * columns, dtype=np.int16).reshape(rows, columns)
        q_codes = ((code_seed + model_index + layer_index) % 15 - 7).astype(np.int8)
        q_col_block_size = 8 if strategy == "Q_block_scale" else None
        if q_col_block_size is None:
            q_scales = np.full(
                (rows,),
                1.125 if strategy == "Q_global_scale" else 1.0,
                dtype=np.float16,
            )
        else:
            q_scales = np.full((rows, columns // q_col_block_size), 1.0, dtype=np.float16)

        sparse_mask = None
        sparse_values = None
        if has_sparse:
            sparse_mask = np.zeros((rows, columns), dtype=bool)
            sparse_values = np.zeros((rows, columns), dtype=np.float32)
            row = layer_index % rows
            column = (3 * layer_index + model_index) % columns
            sparse_mask[row, column] = True
            # OBS changes only the already-paid sparse values; the support and
            # quantized state remain byte-identical to the non-OBS endpoint.
            sparse_values[row, column] = (
                0.25 if "OBS" in strategy else 0.125
            )

        lowrank_left = None
        lowrank_right = None
        if has_lowrank:
            lowrank_left = np.full(
                (rows, rank), 0.03125 * (layer_index + 1), dtype=np.float32
            )
            lowrank_right = np.full(
                (rank, columns), 0.015625 * (model_index + 1), dtype=np.float32
            )

        layers.append(
            LayerCodecPayload(
                name=name,
                q_codes=q_codes,
                q_scales=q_scales,
                q_bits=4,
                q_col_block_size=q_col_block_size,
                sparse_values=sparse_values,
                sparse_mask=sparse_mask,
                lowrank_left=lowrank_left,
                lowrank_right=lowrank_right,
            )
        )
    return layers


def _artifact_record(
    job_dir: Path,
    strategy: str,
    result: ArtifactWriteResult,
    *,
    reference_bytes: int,
    ql_budget_bytes: int,
) -> dict[str, object]:
    capped = strategy in {"Q+S+L_QL_budget", REPORT.STRICT}
    natural_bytes = result.file_bytes - result.tail_padding_bytes
    under_cap: bool | str = natural_bytes <= ql_budget_bytes if capped else "not_applicable"
    return {
        "strategy": strategy,
        "target_ratio": 0.258,
        "artifact_path": result.path.relative_to(job_dir).as_posix(),
        "artifact_sha256": result.sha256,
        "artifact_file_bytes": result.file_bytes,
        "artifact_natural_file_bytes": natural_bytes,
        "artifact_logical_payload_bits": result.logical_payload_bits,
        "artifact_stream_bytes": result.stream_bytes,
        "artifact_container_bytes": result.container_bytes,
        "artifact_alignment_padding_bytes": result.alignment_padding_bytes,
        "artifact_tail_padding_bytes": result.tail_padding_bytes,
        "artifact_total_overhead_bytes": result.file_bytes
        - math.ceil(result.logical_payload_bits / 8.0),
        "reference_artifact_file_bytes": reference_bytes,
        "artifact_to_reference_file_ratio": result.file_bytes / reference_bytes,
        "artifact_physical_compression_ratio": reference_bytes / result.file_bytes,
        "ql_budget_file_bytes": ql_budget_bytes,
        "under_ql_serialized_cap_before_padding": under_cap,
        "same_physical_bytes_as_ql": result.file_bytes == ql_budget_bytes,
        "roundtrip_exact_fp16_endpoint": True,
        "artifact_scope": REPORT.PAYLOAD_SCOPE,
        "production_backend": False,
    }


def _nll_deltas() -> dict[str, float]:
    return {
        "Q": 0.30,
        "Q_global_scale": 0.28,
        "Q_block_scale": 0.20,
        "Q+S": 0.22,
        "Q+S_OBS": 0.18,
        "Q+L": 0.15,
        "Q+S+L_QL_budget": 0.16,
        REPORT.STRICT: 0.14,
        "Q+S+L": 0.13,
        "Q+S_OBS+L": 0.12,
        "Q+S+L_component_scale": 0.11,
    }


def _rho(strategy: str) -> tuple[float, float, float]:
    if strategy in {"Q+S", "Q+S_OBS"}:
        return math.nan, -0.35, math.nan
    if strategy == "Q+L":
        return math.nan, math.nan, -0.50
    if "+S" in strategy and "+L" in strategy:
        return 0.02, -0.30, -0.50
    return math.nan, math.nan, math.nan


def _geometry(strategy: str) -> dict[str, float]:
    rho_sl, rho_qs, rho_ql = _rho(strategy)
    self_q = 1.0
    self_s = 0.2 if "+S" in strategy else 0.0
    self_l = 0.3 if "+L" in strategy else 0.0

    def cross(rho: float, left: float, right: float) -> float:
        if math.isnan(rho):
            return 0.0
        return rho * math.sqrt(2.0 * left * 2.0 * right)

    cross_qs = cross(rho_qs, self_q, self_s)
    cross_ql = cross(rho_ql, self_q, self_l)
    cross_sl = cross(rho_sl, self_s, self_l)
    return {
        "hessian_self_q": self_q,
        "hessian_self_s": self_s,
        "hessian_self_l": self_l,
        "hessian_cross_qs": cross_qs,
        "hessian_cross_ql": cross_ql,
        "hessian_cross_sl": cross_sl,
        "hessian_cost": self_q + self_s + self_l + cross_qs + cross_ql + cross_sl,
    }


def _selected_names(stage: dict[str, object]) -> list[str]:
    scope = stage["tensor_scope"]
    assert isinstance(scope, dict)
    return [
        f"model.layers.{layer}.mlp.{module}"
        for layer in scope["layers"]
        for module in scope["module_types"]
    ]


def _window_rows(baseline_windows: list[float]) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    rows: list[dict[str, object]] = []
    aggregates: dict[str, dict[str, float]] = {}
    for strategy in ("dense", *REPORT.EXPECTED_STRATEGIES):
        delta = 0.0 if strategy == "dense" else _nll_deltas()[strategy]
        values = [value + delta for value in baseline_windows]
        for index, nll in enumerate(values):
            rows.append(
                {
                    "batch_index": index,
                    "nll": nll,
                    "nll_sum": nll * 10,
                    "perplexity": math.exp(nll),
                    "sequence_index": 0,
                    "strategy": strategy,
                    "tokens": 10,
                    "window_index": index,
                }
            )
        aggregates[strategy] = {
            "nll": sum(value * 10 for value in values) / 80,
            "perplexity": math.exp(sum(value * 10 for value in values) / 80),
        }
    return rows, aggregates


def _paired(values: list[float]) -> tuple[float, float, float, float]:
    mean = REPORT.statistics.fmean(values)
    se = REPORT.statistics.stdev(values) / math.sqrt(len(values))
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


def _make_fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    source_config = json.loads(
        (REPO_ROOT / "configs" / "large_scale_hessian_pilot_20260714.json").read_text(
            encoding="utf-8"
        )
    )
    source_config["output_root"] = "results/large_scale_hessian_pilot_20260714"
    source_config["runner"] = "scripts/run_pretrained_hessian_repair.py"
    config_dir = repo / "configs"
    config_dir.mkdir()
    config_path = config_dir / "large_scale_hessian_pilot_20260714.json"
    config_path.write_text(
        json.dumps(source_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    config_sha = _sha(config_path)

    source_files = {
        "runner": "scripts/run_pretrained_hessian_repair.py",
        "codec": "src/llm_spectral_dynamics/structured/codec_artifact.py",
        "hessian_repair": "src/llm_spectral_dynamics/structured/hessian_repair.py",
        "base_runner": "scripts/run_pretrained_llm_orthogonality.py",
    }
    source_snapshot: dict[str, dict[str, object]] = {}
    for name, relative in source_files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# deterministic {name} fixture\n", encoding="utf-8")
        source_snapshot[name] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
    source_sha = REPORT._object_sha256(source_snapshot)
    suite_root = repo / source_config["output_root"]
    (suite_root / "jobs").mkdir(parents=True)
    (suite_root / "_state").mkdir()
    (suite_root / "_logs").mkdir()

    entries: list[dict[str, object]] = []
    for model_index, stage in enumerate(source_config["stages"]):
        stage_id = stage["id"]
        job_id = REPORT._job_id(stage_id)
        job_dir = suite_root / "jobs" / job_id
        artifacts_dir = job_dir / "artifacts"
        figures_dir = job_dir / "figures"
        artifacts_dir.mkdir(parents=True)
        figures_dir.mkdir()
        selected = _selected_names(stage)
        selected_parameters = len(selected) * 16 * 32
        model_parameters = 5 * selected_parameters
        baseline_windows = [
            3.90 + 0.10 * model_index,
            4.10 + 0.10 * model_index,
            4.00 + 0.10 * model_index,
            4.20 + 0.10 * model_index,
            3.80 + 0.10 * model_index,
            4.30 + 0.10 * model_index,
            3.70 + 0.10 * model_index,
            4.00 + 0.10 * model_index,
        ]
        window_rows, aggregates = _window_rows(baseline_windows)
        baseline_nll = aggregates["dense"]["nll"]
        baseline_ppl = aggregates["dense"]["perplexity"]

        model_argument = stage["model"]
        entry: dict[str, object] = {
            "job_id": job_id,
            "stage_id": stage_id,
            "lane": stage["lane"],
            "evidence_role": REPORT.EVIDENCE_ROLE,
            "protocol_manifest_consumed": False,
            "seed_aggregation_allowed": False,
            "data_window_independence": "shared_sequential_windows_not_independent_across_seeds",
            "model_declared": stage["model"],
            "model_argument": model_argument,
            "model_scale": stage["model_scale"],
            "model_availability": stage["model_availability"],
            "availability_note": stage["availability_note"],
            "model_override_env": stage["model_override_env"],
            "revision": stage["revision"] or None,
            "seed": 17,
            "target_rate": 0.258,
            "tensor_scope": stage["tensor_scope"],
            "suite_config_sha256": config_sha,
            "numerical_source_sha256": source_sha,
            "status": "completed_valid",
            "reason": None,
            "exit_code": 0,
        }
        job_sha = REPORT._expected_job_hash(source_config, stage, entry)
        fingerprint = REPORT._object_sha256(
            {
                "suite_config_sha256": config_sha,
                "job_config_sha256": job_sha,
                "numerical_source_sha256": source_sha,
            }
        )
        entry["job_config_sha256"] = job_sha
        entry["execution_fingerprint_sha256"] = fingerprint

        arguments = dict(source_config["common"])
        arguments.update(
            {
                "model": model_argument,
                "revision": stage["revision"],
                "module_types": stage["tensor_scope"]["module_types"],
                "layers": stage["tensor_scope"]["layers"],
                "max_modules": stage["tensor_scope"]["max_modules"],
                "target_ratios": [0.258],
                "endpoint_target": 0.258,
                "seed": 17,
                "device": "cuda",
                "svd_device": "cuda",
                "selected_layers": selected,
            }
        )
        run_config = {
            "model": model_argument,
            "revision": stage["revision"],
            "model_identity": {
                "resolved_model_commit_hash": REPORT.EXPECTED_STAGE_MATERIAL[stage_id][
                    "resolved_commit"
                ],
                "model_name_or_path": model_argument,
            },
            "source_snapshot": source_snapshot,
            "selected_layers": selected,
            "selected_parameter_count": selected_parameters,
            "model_parameter_count": model_parameters,
            "baseline_metrics": {
                "nll": baseline_nll,
                "perplexity": baseline_ppl,
                "tokens": 80,
            },
            "actual_eval_tokens": 80,
            "payload_scope": REPORT.PAYLOAD_SCOPE,
            "data": {
                "requested": {
                    "dataset": "wikitext",
                    "subset": "wikitext-2-raw-v1",
                    "split": "validation",
                    "backup_name": "",
                    "sequence_length": 128,
                    "batch_size": 1,
                    "allow_fallback": False,
                },
                "source_used": "dataset:wikitext",
                "source_metadata": [
                    {
                        "source": "dataset",
                        "dataset": "wikitext",
                        "subset": "wikitext-2-raw-v1",
                        "split": "validation",
                        "backup_name": "",
                        "rows_requested": 168,
                    }
                ],
                "fallback_allowed": False,
                "text_pool_count": 168,
                "unique_text_pool_count": 104,
                "calib_text_count": 32,
                "eval_text_count": 64,
                "eval_window_count": 8,
                "content_disjoint": True,
                "identical_text_overlap_count": 0,
                "split_policy": "content_disjoint_sequential_text_windows",
                "window_interval_semantics": "paired fixed-window mean +/- 1.96 standard errors; descriptive, not an independence-based population CI",
                "calib_digest": hashlib.sha256(
                    f"fixture-calibration-{model_index}".encode("utf-8")
                ).hexdigest(),
                "eval_digest": hashlib.sha256(
                    f"fixture-evaluation-{model_index}".encode("utf-8")
                ).hexdigest(),
            },
            "activation_counts": {name: 32 for name in selected},
            "runtime": {
                "python": "3.10",
                "platform": "fixture",
                "torch": "fixture",
                "transformers": "fixture",
                "datasets": "fixture",
                "numpy": "fixture",
                "cuda_available": True,
                "cuda_device": "fixture GPU",
            },
            "arguments": arguments,
        }
        # The real pilot evaluates 8 * (128 - 1) tokens. Keep that invariant in
        # the fixture while retaining 8 compact window rows by using 127 tokens.
        for row in window_rows:
            row["tokens"] = 127
            row["nll_sum"] = float(row["nll"]) * 127
        run_config["actual_eval_tokens"] = 1016
        run_config["baseline_metrics"]["tokens"] = 1016

        reference = artifacts_dir / "selected_linear_fp16_reference.hrc"
        reference_weights = {
            name: np.full(
                (16, 32),
                0.01 * (model_index + layer_index + 1),
                dtype=np.float32,
            )
            for layer_index, name in enumerate(selected)
        }
        reference_result = write_fp16_reference_artifact(
            reference, reference_weights, alignment=REPORT.CODEC_ALIGNMENT
        )
        reference_record = {
            "path": reference.relative_to(job_dir).as_posix(),
            "sha256": reference_result.sha256,
            "file_bytes": reference_result.file_bytes,
            "logical_payload_bits": reference_result.logical_payload_bits,
            "roundtrip_exact_fp16": True,
        }

        payloads = {
            strategy: _codec_layers(strategy, selected, model_index)
            for strategy in REPORT.EXPECTED_STRATEGIES
        }
        ql_path = artifacts_dir / "Q_L.hrc"
        ql_result = write_codec_artifact(
            ql_path,
            payloads["Q+L"],
            alignment=REPORT.CODEC_ALIGNMENT,
        )
        ql_budget_bytes = ql_result.file_bytes
        artifact_records: list[dict[str, object]] = []
        endpoint_rows: list[dict[str, object]] = []
        for strategy in REPORT.EXPECTED_STRATEGIES:
            artifact = artifacts_dir / (strategy.replace("+", "_") + ".hrc")
            if strategy == "Q+L":
                result = ql_result
            else:
                result = write_codec_artifact(
                    artifact,
                    payloads[strategy],
                    alignment=REPORT.CODEC_ALIGNMENT,
                    target_file_bytes=(
                        ql_budget_bytes
                        if strategy in {"Q+S+L_QL_budget", REPORT.STRICT}
                        else None
                    ),
                )
            record = _artifact_record(
                job_dir,
                strategy,
                result,
                reference_bytes=reference_result.file_bytes,
                ql_budget_bytes=ql_budget_bytes,
            )
            artifact_records.append(record)
            delta = _nll_deltas()[strategy]
            aggregate_nll = baseline_nll + delta
            aggregate_ppl = math.exp(aggregate_nll)
            dense_values = [float(row["nll"]) for row in window_rows if row["strategy"] == "dense"]
            strategy_values = [float(row["nll"]) for row in window_rows if row["strategy"] == strategy]
            paired = _paired([left - right for left, right in zip(strategy_values, dense_values)])
            rho_sl, rho_qs, rho_ql = _rho(strategy)
            geometry = _geometry(strategy)
            strategy_payloads = payloads[strategy]
            sparse_nnz = sum(
                int(np.count_nonzero(layer.sparse_mask))
                if layer.sparse_mask is not None
                else 0
                for layer in strategy_payloads
            )
            lowrank_rank_sum = sum(
                int(np.asarray(layer.lowrank_left).shape[1])
                if layer.lowrank_left is not None
                else 0
                for layer in strategy_payloads
            )
            layers_s_active = sum(
                layer.sparse_mask is not None
                and int(np.count_nonzero(layer.sparse_mask)) > 0
                for layer in strategy_payloads
            )
            layers_l_active = sum(
                layer.lowrank_left is not None
                and int(np.asarray(layer.lowrank_left).shape[1]) > 0
                for layer in strategy_payloads
            )
            layers_both_active = sum(
                layer.sparse_mask is not None
                and int(np.count_nonzero(layer.sparse_mask)) > 0
                and layer.lowrank_left is not None
                and int(np.asarray(layer.lowrank_left).shape[1]) > 0
                for layer in strategy_payloads
            )
            q_scale_count = sum(
                int(np.asarray(layer.q_scales).size) for layer in strategy_payloads
            )
            if strategy == "Q_global_scale":
                folded_repair_dof = len(strategy_payloads)
            elif strategy == "Q_block_scale":
                folded_repair_dof = q_scale_count
            elif strategy in {"Q+S_OBS", "Q+S_OBS+L"}:
                folded_repair_dof = sparse_nnz
            elif strategy in {REPORT.STRICT, "Q+S+L_component_scale"}:
                folded_repair_dof = (
                    len(strategy_payloads) + layers_s_active + layers_l_active
                )
            else:
                folded_repair_dof = 0
            endpoint_rows.append(
                {
                    **record,
                    "heldout_evaluated": True,
                    "heldout_tokens": 1016,
                    "heldout_nll": aggregate_nll,
                    "heldout_perplexity": aggregate_ppl,
                    "nll_delta": aggregate_nll - baseline_nll,
                    "perplexity_delta": aggregate_ppl - baseline_ppl,
                    "paired_window_count": 8,
                    "paired_window_nll_delta_mean": paired[0],
                    "paired_window_nll_delta_se": paired[1],
                    "paired_window_nll_delta_ci95_low": paired[2],
                    "paired_window_nll_delta_ci95_high": paired[3],
                    "normalized_hessian_cost": 2.0 * geometry["hessian_cost"] / 200.0,
                    "hessian_cost": geometry["hessian_cost"],
                    "baseline_hessian_energy": 200.0,
                    "hessian_self_q": geometry["hessian_self_q"],
                    "hessian_self_s": geometry["hessian_self_s"],
                    "hessian_self_l": geometry["hessian_self_l"],
                    "hessian_cross_qs": geometry["hessian_cross_qs"],
                    "hessian_cross_ql": geometry["hessian_cross_ql"],
                    "hessian_cross_sl": geometry["hessian_cross_sl"],
                    "artifact_scope": REPORT.PAYLOAD_SCOPE,
                    "production_backend": False,
                    "roundtrip_exact_fp16_endpoint": True,
                    "rate_cap_satisfied": (
                        record["under_ql_serialized_cap_before_padding"]
                        if strategy in {"Q+S+L_QL_budget", REPORT.STRICT}
                        else "not_applicable"
                    ),
                    "rho_sl": rho_sl,
                    "rho_qs": rho_qs,
                    "rho_ql": rho_ql,
                    "rho_sl_kind": _rho_kind(rho_sl),
                    "rho_qs_kind": _rho_kind(rho_qs),
                    "rho_ql_kind": _rho_kind(rho_ql),
                    "q_scale_count": q_scale_count,
                    "sparse_nnz": sparse_nnz,
                    "lowrank_rank_sum": lowrank_rank_sum,
                    "layers_s_active": layers_s_active,
                    "layers_l_active": layers_l_active,
                    "layers_both_s_l_active": layers_both_active,
                    "folded_repair_dof": folded_repair_dof,
                }
            )
        artifact_manifest = {
            "format": "llm_spectral_dynamics_research_codec",
            "scope": REPORT.PAYLOAD_SCOPE,
            "production_backend": False,
            "alignment_bytes": REPORT.CODEC_ALIGNMENT,
            "serialized_rate_cap_enforced": True,
            "rate_cap_policy": "fixture mirrors the exact Q+L serialized cap",
            "reference": reference_record,
            "strategies": artifact_records,
        }
        (job_dir / "artifact_manifest.json").write_text(
            json.dumps(artifact_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        _write_csv(job_dir / "artifact_payloads.csv", artifact_records)
        _write_csv(job_dir / "strategy_endpoints.csv", endpoint_rows)
        _write_csv(job_dir / "endpoint_window_nll.csv", window_rows)

        comfort_rows: list[dict[str, object]] = []
        comfort_summaries: list[dict[str, object]] = []
        epsilons = [float(value) for value in source_config["common"]["comfort_epsilons"]]
        endpoint_by_strategy = {row["strategy"]: row for row in endpoint_rows}
        for strategy in source_config["common"]["comfort_strategies"]:
            endpoint_delta = float(endpoint_by_strategy[strategy]["nll_delta"])
            linear = 0.2 * endpoint_delta
            quadratic = 0.8 * endpoint_delta
            proxy_values: list[float] = []
            task_values: list[float] = []
            for epsilon in epsilons:
                delta = linear * epsilon + quadratic * epsilon * epsilon
                proxy = float(endpoint_by_strategy[strategy]["normalized_hessian_cost"]) * epsilon * epsilon
                nll = baseline_nll + delta
                proxy_values.append(proxy)
                task_values.append(delta)
                comfort_rows.append(
                    {
                        "comfort_tolerance": 1e-4,
                        "deployable": epsilon == 1.0,
                        "epsilon": epsilon,
                        "hessian_cost": float(endpoint_by_strategy[strategy]["hessian_cost"]) * epsilon * epsilon,
                        "nll": nll,
                        "nll_delta": nll - baseline_nll,
                        "normalized_hessian_cost": proxy,
                        "path_kind": "codec_endpoint" if epsilon == 1.0 else "noncodec_interpolation",
                        "payload_ratio_at_codec_endpoint": 0.258,
                        "perplexity": math.exp(nll),
                        "perplexity_delta": math.exp(nll) - baseline_ppl,
                        "strategy": strategy,
                        "target_ratio": 0.258,
                        "taylor_fit_absolute_error": 0.0,
                        "taylor_fit_nll_delta": delta,
                        "tokens": 1016,
                        "within_local_comfort_fit": True,
                    }
                )
            comfort_summaries.append(
                {
                    "codec_endpoint_fit_error": 0.0,
                    "codec_endpoint_nll_delta": endpoint_delta,
                    "codec_endpoint_within_comfort_fit": True,
                    "hessian_proxy_nll_correlation": REPORT._pearson(proxy_values, task_values),
                    "interpretation": "local_fit_reaches_codec_endpoint",
                    "max_contiguous_comfort_epsilon": 1.0,
                    "small_epsilon_fit_max": 0.125,
                    "strategy": strategy,
                    "target_ratio": 0.258,
                    "taylor_linear_coefficient": linear,
                    "taylor_quadratic_coefficient": quadratic,
                }
            )
        _write_csv(job_dir / "comfort_sweep.csv", comfort_rows)
        _write_csv(job_dir / "comfort_summary.csv", comfort_summaries)

        psd_rows: list[dict[str, object]] = []
        negative_relatives: list[float] = []
        shift_relatives: list[float] = []
        for layer_index, layer in enumerate(selected):
            original_scale = 1.0 + 0.01 * layer_index
            original_min = -(layer_index + 1) * 1e-10
            original_relative = original_min / original_scale
            diagonal_shift = -original_min + REPORT.FLOAT32_PSD_FLOOR_RTOL * original_scale
            shift_relative = diagonal_shift / original_scale
            psd_rows.append(
                {
                    "layer": layer,
                    "original_min_eigenvalue": original_min,
                    "original_spectral_scale": original_scale,
                    "original_min_relative": original_relative,
                    "diagonal_shift": diagonal_shift,
                    "diagonal_shift_relative": shift_relative,
                    "final_min_eigenvalue": original_min + diagonal_shift,
                    "final_spectral_scale": original_scale + diagonal_shift,
                    "repair_applied": True,
                    "psd_rejection_rtol": REPORT.PSD_REJECTION_RTOL,
                    "float32_storage_floor_rtol": REPORT.FLOAT32_PSD_FLOOR_RTOL,
                }
            )
            negative_relatives.append(max(0.0, -original_relative))
            shift_relatives.append(shift_relative)
        _write_csv(job_dir / "covariance_psd_audit.csv", psd_rows)
        run_config["covariance_psd_audit"] = {
            "path": "covariance_psd_audit.csv",
            "layer_count": len(psd_rows),
            "psd_rejection_rtol": REPORT.PSD_REJECTION_RTOL,
            "float32_storage_floor_rtol": REPORT.FLOAT32_PSD_FLOOR_RTOL,
            "maximum_original_negative_relative": max(negative_relatives),
            "maximum_diagonal_shift_relative": max(shift_relatives),
            "all_consumers_share_prepared_covariance": True,
        }
        (job_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for relative in source_config["expected_outputs"]:
            path = job_dir / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture evidence\n", encoding="utf-8")
        (job_dir / "COMPLETED").write_text('{"completed": true}\n', encoding="utf-8")

        evidence = REPORT._runner_evidence(stage, run_config, artifact_manifest)
        evidence_sha = REPORT._object_sha256(evidence)
        stdout = suite_root / "_logs" / f"{job_id}.stdout.log"
        stderr = suite_root / "_logs" / f"{job_id}.stderr.log"
        stdout.write_text("fixture stdout\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        record = {
            "schema_version": REPORT.JOB_RECORD_SCHEMA,
            "job_id": job_id,
            "suite_id": REPORT.SUITE_ID,
            "evidence_role": REPORT.EVIDENCE_ROLE,
            "protocol_manifest_consumed": False,
            "seed_aggregation_allowed": False,
            "suite_config_sha256": config_sha,
            "job_config_sha256": job_sha,
            "numerical_source_sha256": source_sha,
            "execution_fingerprint_sha256": fingerprint,
            "status": "COMPLETED",
            "exit_code": 0,
            "evidence_sha256": evidence_sha,
            "stdout": {
                "path": stdout.relative_to(suite_root).as_posix(),
                "size_bytes": stdout.stat().st_size,
                "sha256": _sha(stdout),
            },
            "stderr": {
                "path": stderr.relative_to(suite_root).as_posix(),
                "size_bytes": stderr.stat().st_size,
                "sha256": _sha(stderr),
            },
        }
        (job_dir / "_suite_job_record.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        (suite_root / "_state" / f"{job_id}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        entry["actual"] = evidence
        entries.append(entry)

    suite_manifest = {
        "schema_version": REPORT.SUITE_MANIFEST_SCHEMA,
        "suite_id": REPORT.SUITE_ID,
        "suite_config_sha256": config_sha,
        "numerical_source_snapshot": source_snapshot,
        "method_contract": {"expected_strategies": list(REPORT.EXPECTED_STRATEGIES)},
        "status_counts": {
            "completed_valid": 3,
            "failed": 0,
            "invalid": 0,
            "planned": 0,
            "running": 0,
        },
        "jobs": entries,
    }
    (suite_root / "suite_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "repo": repo,
        "config": config_path,
        "suite": suite_root,
        "output": repo / "paper" / "results",
    }


@pytest.fixture
def pilot(tmp_path: Path) -> dict[str, Path]:
    return _make_fixture(tmp_path)


def _build(pilot: dict[str, Path]) -> list[Path]:
    return REPORT.build_report(
        pilot["config"], pilot["suite"], pilot["output"], repo_root=pilot["repo"]
    )


def _rewrite_artifact_record(
    job: Path, strategy: str, updates: dict[str, object]
) -> dict[str, object]:
    """Keep all outer ledgers synchronized so container checks are exercised."""

    manifest_path = job / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["strategies"] if item["strategy"] == strategy)
    record.update(updates)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    payload_path = job / "artifact_payloads.csv"
    payload_rows = list(csv.DictReader(payload_path.open("r", encoding="utf-8", newline="")))
    payload_row = next(item for item in payload_rows if item["strategy"] == strategy)
    payload_row.update(record)
    _write_csv(payload_path, payload_rows)

    endpoint_path = job / "strategy_endpoints.csv"
    endpoint_rows = list(csv.DictReader(endpoint_path.open("r", encoding="utf-8", newline="")))
    endpoint_row = next(item for item in endpoint_rows if item["strategy"] == strategy)
    endpoint_row.update({key: value for key, value in record.items() if key in endpoint_row})
    _write_csv(endpoint_path, endpoint_rows)
    return record


def _internal_padding_offset(raw: bytes) -> int:
    _, _, alignment, header_size = REPORT.CODEC_PREFIX.unpack_from(raw)
    header_end = REPORT.CODEC_PREFIX.size + int(header_size)
    payload_base = (header_end + int(alignment) - 1) // int(alignment) * int(alignment)
    header = json.loads(raw[REPORT.CODEC_PREFIX.size : header_end].decode("utf-8"))
    previous_end = 0
    for stream in header["streams"]:
        offset = int(stream["offset"])
        if offset > previous_end:
            return payload_base + previous_end
        previous_end = offset + int(stream["nbytes"])
    raise AssertionError("fixture HRC unexpectedly has no internal alignment padding")


def test_valid_three_job_fixture_emits_deterministic_unpooled_outputs(pilot: dict[str, Path]) -> None:
    paths = _build(pilot)
    assert {path.name for path in paths} == {
        "scaling_pilot_endpoints.csv",
        "scaling_pilot_pairs.csv",
        "scaling_pilot_models.csv",
        "scaling_pilot_paths.csv",
        "scaling_pilot_table.tex",
        "scaling_pilot_endpoints_table.tex",
        "scaling_pilot_numbers.tex",
        "scaling_pilot_summary.md",
        "scaling_pilot_manifest.json",
    }
    first = {path.name: _sha(path) for path in paths}
    second = {path.name: _sha(path) for path in _build(pilot)}
    assert first == second
    with (pilot["output"] / "scaling_pilot_endpoints.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        endpoints = list(csv.DictReader(handle))
    with (pilot["output"] / "scaling_pilot_pairs.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        pairs = list(csv.DictReader(handle))
    with (pilot["output"] / "scaling_pilot_models.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        models = list(csv.DictReader(handle))
    with (pilot["output"] / "scaling_pilot_paths.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        path_rows = list(csv.DictReader(handle))
    assert len(endpoints) == 33
    assert len(pairs) == 12
    assert len(models) == 3
    assert len(path_rows) == 3 * 6 * 13
    figure_paths = [
        row for row in path_rows if row["strategy"] in {"Q+L", REPORT.STRICT}
    ]
    assert len(figure_paths) == 3 * 2 * 13
    assert all(row["fit_is_extrapolation"] == str(float(row["epsilon"]) > 0.125).lower() for row in path_rows)
    assert all(float(row["strict_minus_ql_nll"]) < 0 for row in models)
    manifest = json.loads(
        (pilot["output"] / "scaling_pilot_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["observation_count"] == 3
    assert "cross-model mean" in manifest["claim_limitations"][0]
    assert all(not Path(key).is_absolute() for key in manifest["input_sha256"])
    summary = (pilot["output"] / "scaling_pilot_summary.md").read_text(encoding="utf-8")
    normalized_summary = " ".join(summary.split())
    assert "not a multi-seed result" in normalized_summary
    assert "fixed windows and is not a significance test" in normalized_summary
    assert "Literature numbers are not placed on these axes" in normalized_summary


@pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")
def test_scaling_plots_close_manifest_and_are_byte_deterministic(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _build(pilot)
    spec = importlib.util.spec_from_file_location(
        "scaling_pilot_plots_test",
        REPO_ROOT / "paper" / "figures" / "scaling_pilot_plots.py",
    )
    assert spec is not None and spec.loader is not None
    plots = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plots
    spec.loader.exec_module(plots)

    figure_dir = pilot["repo"] / "paper" / "figures"
    figure_dir.mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "paper" / "fonts", pilot["repo"] / "paper" / "fonts")
    script_copy = figure_dir / "scaling_pilot_plots.py"
    shutil.copy2(REPO_ROOT / "paper" / "figures" / "scaling_pilot_plots.py", script_copy)
    monkeypatch.setattr(plots, "ROOT", pilot["repo"])
    monkeypatch.setattr(plots, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(plots, "RESULT_DIR", pilot["output"])
    monkeypatch.setattr(plots, "ENDPOINTS", pilot["output"] / "scaling_pilot_endpoints.csv")
    monkeypatch.setattr(plots, "PAIRS", pilot["output"] / "scaling_pilot_pairs.csv")
    monkeypatch.setattr(plots, "MODELS", pilot["output"] / "scaling_pilot_models.csv")
    monkeypatch.setattr(plots, "PATHS", pilot["output"] / "scaling_pilot_paths.csv")
    monkeypatch.setattr(
        plots, "REPORT_MANIFEST", pilot["output"] / "scaling_pilot_manifest.json"
    )
    monkeypatch.setattr(plots, "__file__", str(script_copy))

    plots.main()
    generated = sorted(
        path
        for path in figure_dir.glob("scaling_pilot_*")
        if path.suffix in {".pdf", ".svg", ".png", ".json"}
    )
    assert len(generated) == 8
    first = {path.name: _sha(path) for path in generated}
    time.sleep(1.1)
    plots.main()
    second = {path.name: _sha(path) for path in generated}
    assert first == second
    for stem in ("scaling_pilot_efficiency_geometry", "scaling_pilot_loss_paths"):
        manifest = json.loads(
            (figure_dir / f"{stem}.manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["schema_version"] == plots.FIGURE_SCHEMA
        assert manifest["evidence_status"] == plots.EVIDENCE_STATUS
        for name, digest in manifest["outputs"].items():
            assert _sha(figure_dir / name) == digest

    endpoints = pilot["output"] / "scaling_pilot_endpoints.csv"
    endpoints.write_bytes(endpoints.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="does not match scaling report manifest"):
        plots.main()


def test_missing_endpoint_strategy_fails_closed(pilot: dict[str, Path]) -> None:
    path = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[0]) / "strategy_endpoints.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    _write_csv(path, rows[:-1])
    with pytest.raises(REPORT.ReportError, match="endpoint strategy order/set"):
        _build(pilot)


def test_strict_and_ql_final_bytes_must_be_equal(pilot: dict[str, Path]) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[0])
    manifest_path = job / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strict = next(item for item in manifest["strategies"] if item["strategy"] == REPORT.STRICT)
    artifact = job / strict["artifact_path"]
    artifact.write_bytes(artifact.read_bytes() + b"\0")
    file_bytes = int(strict["artifact_file_bytes"]) + 1
    reference_bytes = int(strict["reference_artifact_file_bytes"])
    _rewrite_artifact_record(
        job,
        REPORT.STRICT,
        {
            "artifact_file_bytes": file_bytes,
            "artifact_tail_padding_bytes": int(strict["artifact_tail_padding_bytes"]) + 1,
            "artifact_total_overhead_bytes": int(strict["artifact_total_overhead_bytes"]) + 1,
            "artifact_sha256": _sha(artifact),
            "artifact_to_reference_file_ratio": file_bytes / reference_bytes,
            "artifact_physical_compression_ratio": reference_bytes / file_bytes,
            "same_physical_bytes_as_ql": False,
        },
    )
    with pytest.raises(REPORT.ReportError, match=r"strict QSL and Q\+L final artifact bytes differ"):
        _build(pilot)


def test_artifact_sha_tampering_fails_closed(pilot: dict[str, Path]) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[1])
    manifest = json.loads((job / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact = job / manifest["strategies"][0]["artifact_path"]
    raw = bytearray(artifact.read_bytes())
    raw[-1] ^= 1
    artifact.write_bytes(raw)
    with pytest.raises(REPORT.ReportError, match="artifact SHA-256 mismatch"):
        _build(pilot)


@pytest.mark.parametrize(
    ("padding_kind", "strategy"),
    (("internal", "Q+L"), ("tail", REPORT.STRICT)),
)
def test_hrc_padding_tampering_fails_even_with_outer_sha_synchronized(
    pilot: dict[str, Path], padding_kind: str, strategy: str
) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[0])
    manifest = json.loads((job / "artifact_manifest.json").read_text(encoding="utf-8"))
    record = next(item for item in manifest["strategies"] if item["strategy"] == strategy)
    artifact = job / record["artifact_path"]
    raw = bytearray(artifact.read_bytes())
    if padding_kind == "internal":
        offset = _internal_padding_offset(raw)
    else:
        assert int(record["artifact_tail_padding_bytes"]) > 0
        offset = len(raw) - 1
    assert raw[offset] == 0
    raw[offset] = 1
    artifact.write_bytes(raw)
    _rewrite_artifact_record(job, strategy, {"artifact_sha256": _sha(artifact)})

    with pytest.raises(REPORT.ReportError, match="codec padding"):
        _build(pilot)


def test_artifact_payload_semantic_tampering_fails_closed(pilot: dict[str, Path]) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[1])
    path = job / "artifact_payloads.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    row = next(item for item in rows if item["strategy"] == "Q_block_scale")
    row["artifact_stream_bytes"] = str(int(row["artifact_stream_bytes"]) + 1)
    _write_csv(path, rows)
    with pytest.raises(REPORT.ReportError, match="payload CSV artifact_stream_bytes mismatch"):
        _build(pilot)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_layer", "PSD rows do not exactly cover selected tensors"),
        ("shift_tamper", "PSD shift exceeds the declared float32 floor repair"),
    ),
)
def test_covariance_psd_audit_tampering_fails_closed(
    pilot: dict[str, Path], mutation: str, message: str
) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[2])
    path = job / "covariance_psd_audit.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if mutation == "missing_layer":
        rows = rows[:-1]
    else:
        rows[0]["diagonal_shift"] = str(float(rows[0]["diagonal_shift"]) + 1e-3)
        rows[0]["diagonal_shift_relative"] = str(
            float(rows[0]["diagonal_shift_relative"]) + 1e-3
        )
        rows[0]["final_min_eigenvalue"] = str(
            float(rows[0]["final_min_eigenvalue"]) + 1e-3
        )
        rows[0]["final_spectral_scale"] = str(
            float(rows[0]["final_spectral_scale"]) + 1e-3
        )
    _write_csv(path, rows)
    with pytest.raises(REPORT.ReportError, match=message):
        _build(pilot)


def test_duplicate_fixed_window_fails_closed(pilot: dict[str, Path]) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[2])
    path = job / "endpoint_window_nll.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    strategy_rows = [row for row in rows if row["strategy"] == REPORT.STRICT]
    strategy_rows[-1]["window_index"] = strategy_rows[-2]["window_index"]
    _write_csv(path, rows)
    with pytest.raises(REPORT.ReportError, match="duplicate"):
        _build(pilot)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("token_reallocation", "declared 127 predicted tokens"),
        ("identity", "batch/sequence identity"),
    ),
)
def test_fixed_window_tokens_and_identity_are_closed(
    pilot: dict[str, Path], mutation: str, message: str
) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[0])
    path = job / "endpoint_window_nll.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if mutation == "token_reallocation":
        for row in rows:
            if row["window_index"] == "0":
                row["tokens"] = "126"
            elif row["window_index"] == "1":
                row["tokens"] = "128"
            row["nll_sum"] = str(float(row["nll"]) * int(row["tokens"]))
    else:
        rows[0]["batch_index"] = "999"
        rows[0]["sequence_index"] = "999"
    _write_csv(path, rows)
    with pytest.raises(REPORT.ReportError, match=message):
        _build(pilot)


def test_comfort_proxy_path_must_terminate_at_endpoint(pilot: dict[str, Path]) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[1])
    path = job / "comfort_sweep.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    for row in rows:
        row["normalized_hessian_cost"] = str(2.0 * float(row["normalized_hessian_cost"]))
    _write_csv(path, rows)
    with pytest.raises(REPORT.ReportError, match="path normalized Hessian"):
        _build(pilot)


def test_taylor_absolute_error_is_recomputed(pilot: dict[str, Path]) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[0])
    path = job / "comfort_sweep.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    rows[7]["taylor_fit_absolute_error"] = "999"
    _write_csv(path, rows)
    with pytest.raises(REPORT.ReportError, match="Taylor absolute error"):
        _build(pilot)


def test_endpoint_psd_proxy_formula_is_recomputed(pilot: dict[str, Path]) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[2])
    path = job / "strategy_endpoints.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    q = next(row for row in rows if row["strategy"] == "Q")
    q["normalized_hessian_cost"] = "999"
    q["hessian_cost"] = "-123"
    q["baseline_hessian_energy"] = "-1"
    _write_csv(path, rows)
    with pytest.raises(REPORT.ReportError, match="PSD-proxy energy/cost"):
        _build(pilot)


def test_hessian_decomposition_drift_beyond_float32_bound_fails_closed(
    pilot: dict[str, Path]
) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[2])
    path = job / "strategy_endpoints.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    ql = next(row for row in rows if row["strategy"] == "Q+L")
    ql["hessian_self_l"] = str(float(ql["hessian_self_l"]) + 0.01)
    _write_csv(path, rows)
    with pytest.raises(REPORT.ReportError, match="Hessian self/cross decomposition"):
        _build(pilot)


@pytest.mark.parametrize(("mutation", "message"), (("rho", "rho_sl"), ("counts", "cannot be negative")))
def test_hessian_rho_and_repair_counts_are_recomputed_or_bounded(
    pilot: dict[str, Path], mutation: str, message: str
) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[1])
    path = job / "strategy_endpoints.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    strict = next(row for row in rows if row["strategy"] == REPORT.STRICT)
    if mutation == "rho":
        strict["rho_sl"] = "2.5"
        strict["rho_sl_kind"] = "positive_conflict"
    else:
        strict["folded_repair_dof"] = "-1.0"
    _write_csv(path, rows)
    with pytest.raises(REPORT.ReportError, match=message):
        _build(pilot)


def test_positive_folded_repair_dof_cannot_claim_unstored_state(
    pilot: dict[str, Path]
) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[1])
    path = job / "strategy_endpoints.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    q = next(row for row in rows if row["strategy"] == "Q")
    q["folded_repair_dof"] = "1"
    _write_csv(path, rows)
    with pytest.raises(REPORT.ReportError, match="stored-state reuse"):
        _build(pilot)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("overlap", "data identical_text_overlap_count differs"),
        ("fallback", "dataset fallback was enabled"),
    ),
)
def test_data_overlap_and_fallback_fail_closed(
    pilot: dict[str, Path], mutation: str, message: str
) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[0])
    path = job / "run_config.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "overlap":
        run["data"]["identical_text_overlap_count"] = 1
    else:
        run["data"]["fallback_allowed"] = True
    path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(REPORT.ReportError, match=message):
        _build(pilot)


def test_selected_scope_duplicate_pair_fails_cartesian_closure(
    pilot: dict[str, Path]
) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[1])
    path = job / "run_config.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    first = run["selected_layers"][0]
    first_layer = REPORT._layer_index(first)
    first_module = first.rsplit(".", 1)[-1]
    run["selected_layers"][-1] = f"shadow.layers.{first_layer}.mlp.{first_module}"
    path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(REPORT.ReportError, match="exactly cover layer x module scope"):
        _build(pilot)


def test_candidate_and_report_generator_hashes_are_closed_into_manifest(
    pilot: dict[str, Path]
) -> None:
    _build(pilot)
    manifest_path = pilot["output"] / "scaling_pilot_manifest.json"
    first = json.loads(manifest_path.read_text(encoding="utf-8"))["input_sha256"]
    generator_key = "paper/results/build_scaling_pilot.py"
    assert first[generator_key] == _sha(REPO_ROOT / generator_key)

    job_id = REPORT._job_id(REPORT.EXPECTED_STAGES[2])
    candidate_key = (
        f"results/large_scale_hessian_pilot_20260714/jobs/{job_id}/"
        "candidate_ablation.csv"
    )
    candidate = pilot["repo"] / candidate_key
    assert first[candidate_key] == _sha(candidate)
    candidate.write_text("fixture evidence changed\n", encoding="utf-8")

    _build(pilot)
    second = json.loads(manifest_path.read_text(encoding="utf-8"))["input_sha256"]
    assert second[candidate_key] == _sha(candidate)
    assert second[candidate_key] != first[candidate_key]
    assert second[generator_key] == first[generator_key]


def test_unique_pilot_model_and_scope_contract_is_pinned(pilot: dict[str, Path]) -> None:
    payload = json.loads(pilot["config"].read_text(encoding="utf-8"))
    payload["stages"][0]["model"] = "fabricated/9T"
    payload["stages"][0]["model_scale"] = "9T"
    payload["stages"][0]["tensor_scope"]["layers"] = [90, 91, 92, 93, 94, 95]
    with pytest.raises(REPORT.ReportError, match="material field model"):
        REPORT._validate_config(payload)


def test_job_source_hash_must_match_suite_source(pilot: dict[str, Path]) -> None:
    job = pilot["suite"] / "jobs" / REPORT._job_id(REPORT.EXPECTED_STAGES[0])
    path = job / "run_config.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    run["source_snapshot"]["runner"]["sha256"] = "0" * 64
    path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(REPORT.ReportError, match="run source snapshot differs"):
        _build(pilot)


def test_seed_aggregation_flag_must_remain_false(pilot: dict[str, Path]) -> None:
    path = pilot["suite"] / "suite_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["jobs"][0]["seed_aggregation_allowed"] = True
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(REPORT.ReportError, match="seed_aggregation_allowed mismatch"):
        _build(pilot)
