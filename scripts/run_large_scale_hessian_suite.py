from __future__ import annotations

"""Fail-closed orchestration for staged pretrained Hessian-repair experiments.

The numerical work remains in ``run_pretrained_hessian_repair.py``.  This
module expands a declarative model/seed/tensor-scope/rate matrix, launches one
endpoint rate per process, and records enough evidence to distinguish a real
completed run from an interrupted directory.  It never treats ``RUNNING`` or
``FAILED`` state as resumable.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "large_scale_hessian_suite.v1"
MANIFEST_SCHEMA_VERSION = "large_scale_hessian_suite_manifest.v1"
JOB_RECORD_SCHEMA_VERSION = "large_scale_hessian_job_record.v1"
RESOURCE_GATE_SCHEMA_VERSION = "large_scale_hessian_resource_gate.v1"
RESOURCE_RUNTIME_SCHEMA_VERSION = "large_scale_hessian_resource_runtime.v1"
PROTOCOL_SCHEMA_VERSION = "confirmatory_hessian_protocol.v2"
SHARED_SEQUENTIAL_WINDOW_POLICY = "content-disjoint sequential windows are shared across seed values"
SHARED_SEQUENTIAL_INDEPENDENCE = "shared_sequential_windows_not_independent_across_seeds"
TWO_STAGE_WINDOW_POLICY = (
    "independent train calibration, validation allocation selection, and test endpoint "
    "windows are cross-role content-disjoint"
)
TWO_STAGE_WINDOW_INDEPENDENCE = (
    "independent_train_validation_test_splits_test_reserved_until_after_validation_selection"
)
PROTOCOL_WINDOW_POLICY = (
    "preregistered per-seed calibration source rows are disjoint; "
    "fixed test windows are shared across seed values for paired evaluation"
)
PROTOCOL_WINDOW_INDEPENDENCE = (
    "per_seed_calibration_source_rows_disjoint_fixed_test_windows_shared_for_paired_evaluation"
)
PROTOCOL_ACTIVATION_SAMPLING_POLICY = (
    "deterministic_evenly_spaced_over_all_calibration_token_rows"
)
PRIMARY_RATE_MATCH_RULE = (
    "claim rate-matched only when the candidate and reference artifact files have exactly equal "
    "actual serialized bytes"
)
CO_PRIMARY_RATE_COMPARISON_RULE = (
    "actual_serialized_rate_pareto_only_not_a_rate_matched_contrast"
)
PROTOCOL_FROZEN_HF_FILE_SHA256 = {
    "model.safetensors": "ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15",
    "tokenizer.json": "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624",
    "config.json": "002050231a9b1ec3ac77aa6b9b3bbdc4d923f4068a7dd33b8da72a9bd6ad9a43",
    "tokenizer_config.json": "70e38394e494931c6f773ba41e19460dd4436526b852207367f04341b4066d3f",
}

RUNNER_ARGUMENT_ORDER = (
    "model",
    "revision",
    "model_snapshot_manifest",
    "model_snapshot_manifest_sha256",
    "model_snapshot_aggregate_sha256",
    "resource_gate_manifest",
    "output_dir",
    "device",
    "svd_device",
    "torch_dtype",
    "local_files_only",
    "dataset",
    "subset",
    "split",
    "calibration_split",
    "selection_split",
    "test_split",
    "protocol_manifest",
    "protocol_manifest_sha256",
    "protocol_seed",
    "protocol_eval_role",
    "backup_name",
    "calib_limit",
    "selection_limit",
    "eval_limit",
    "sequence_length",
    "batch_size",
    "texts_per_batch_window",
    "selector_activation_sample_rows",
    "module_types",
    "layer_positions",
    "layers",
    "max_modules",
    "bits",
    "candidate_bits",
    "candidate_q_group_sizes",
    "candidate_quantizers",
    "candidate_lowrank_factor_bits",
    "candidate_family_top_k",
    "target_ratios",
    "endpoint_target",
    "support_encoding",
    "emit_codec_artifacts",
    "enforce_serialized_rate_cap",
    "artifact_alignment",
    "s_method",
    "l_method",
    "residual_order",
    "covariance_mode",
    "covariance_damping_ratio",
    "whitening_floor_ratio",
    "lowrank_svd_solver",
    "lowrank_svd_oversampling",
    "lowrank_svd_niter",
    "rate_allocation",
    "include_global_single_component_controls",
    "two_stage_selection",
    "selection_top_k",
    "strict_sparse_refit",
    "global_frontier_top_ranks",
    "global_frontier_support_fractions",
    "global_frontier_budget_multipliers",
    "repair_block_sizes",
    "skip_block_scale",
    "max_allocation_ranks",
    "allocation_rank_grid",
    "obs_rcond",
    "scale_min",
    "scale_max",
    "rho_threshold",
    "rate_tolerance",
    "comfort_epsilons",
    "comfort_strategies",
    "comfort_fit_max_epsilon",
    "comfort_relative_tolerance",
    "comfort_absolute_tolerance",
    "skip_comfort",
    "skip_plots",
    "proxy_only",
    "seed",
)
BOOLEAN_RUNNER_ARGUMENTS = {
    "local_files_only",
    "emit_codec_artifacts",
    "enforce_serialized_rate_cap",
    "include_global_single_component_controls",
    "two_stage_selection",
    "skip_block_scale",
    "skip_comfort",
    "skip_plots",
    "proxy_only",
}
PER_JOB_ARGUMENTS = {
    "model",
    "revision",
    "model_snapshot_manifest",
    "model_snapshot_manifest_sha256",
    "model_snapshot_aggregate_sha256",
    "resource_gate_manifest",
    "output_dir",
    "protocol_manifest",
    "protocol_manifest_sha256",
    "protocol_seed",
    "protocol_eval_role",
    "module_types",
    "layer_positions",
    "layers",
    "max_modules",
    "target_ratios",
    "endpoint_target",
    "seed",
}
TRANSFORMED_RUNTIME_ARGUMENTS = {
    "device",
    "svd_device",
    "output_dir",
    "protocol_manifest",
    "resource_gate_manifest",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
LAYER_PATTERNS = (
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.block\.(\d+)\."),
    re.compile(r"\.blocks\.(\d+)\."),
)


class SuiteConfigError(ValueError):
    """The declarative experiment matrix is invalid."""


class EvidenceError(RuntimeError):
    """A result directory cannot support a completed/resumable claim."""


@dataclass(frozen=True)
class SuiteDefinition:
    config_path: Path
    repo_root: Path
    raw: dict[str, Any]
    config_sha256: str
    output_root: Path
    runner: Path
    expected_outputs: tuple[str, ...]
    expected_strategies: tuple[str, ...]
    common: dict[str, Any]
    stages: tuple[dict[str, Any], ...]

    @property
    def suite_id(self) -> str:
        return str(self.raw["suite_id"])


@dataclass(frozen=True)
class SuiteJob:
    suite_id: str
    stage_id: str
    lane: str
    evidence_role: str
    protocol_manifest_consumed: bool
    seed_aggregation_allowed: bool
    data_window_independence: str
    protocol_manifest: str | None
    protocol_manifest_sha256: str | None
    protocol_seed: int | None
    protocol_eval_role: str | None
    model_declared: str
    model_argument: str
    model_scale: str
    model_availability: str
    availability_note: str
    model_override_env: str | None
    revision: str
    seed: int
    target_rate: float
    tensor_scope: dict[str, Any]
    effective_arguments: dict[str, Any]
    output_dir: Path
    expected_outputs: tuple[str, ...]
    expected_strategies: tuple[str, ...]
    suite_config_sha256: str
    job_config_sha256: str
    numerical_source_snapshot: dict[str, dict[str, Any]]
    numerical_source_sha256: str
    execution_fingerprint_sha256: str
    resource_policy: dict[str, Any] | None
    repo_root: Path

    @property
    def job_id(self) -> str:
        rate = f"{self.target_rate:.3f}".replace(".", "p")
        return f"{self.stage_id}__seed{self.seed}__rate{rate}"

    @property
    def state_path(self) -> Path:
        return self.output_dir.parents[1] / "_state" / f"{self.job_id}.json"

    @property
    def stdout_path(self) -> Path:
        return self.output_dir.parents[1] / "_logs" / f"{self.job_id}.stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.output_dir.parents[1] / "_logs" / f"{self.job_id}.stderr.log"

    @property
    def resource_gate_path(self) -> Path:
        return self.output_dir.parents[1] / "_resource" / f"{self.job_id}.gate.json"

    @property
    def resource_runtime_path(self) -> Path:
        return self.output_dir.parents[1] / "_resource" / f"{self.job_id}.runtime.json"


@dataclass(frozen=True)
class JobInspection:
    status: str
    reason: str | None = None
    state: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None


@dataclass
class ResourceLease:
    selected_gpu: int
    lock_path: Path
    descriptor: int
    gate_evidence: dict[str, Any]

    def release(self) -> None:
        try:
            import fcntl

            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)


def _resource_policy_enforced(job: SuiteJob) -> bool:
    return (job.resource_policy or {}).get("enforce_at_runtime") is True


def _gpu_sample(physical_gpu: int) -> dict[str, Any]:
    """Sample one physical GPU by the same index later exposed to CUDA."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical_gpu}",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError(f"cannot sample physical GPU {physical_gpu}: {exc}") from exc
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise EvidenceError(
            f"nvidia-smi returned {len(rows)} rows for physical GPU {physical_gpu}"
        )
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 3:
        raise EvidenceError(f"cannot parse nvidia-smi row for physical GPU {physical_gpu}")
    try:
        observed_gpu, memory_mib, utilization_percent = map(int, fields)
    except ValueError as exc:
        raise EvidenceError(
            f"nvidia-smi returned non-integral values for physical GPU {physical_gpu}"
        ) from exc
    if observed_gpu != physical_gpu:
        raise EvidenceError(
            f"nvidia-smi physical index differs: {observed_gpu} != {physical_gpu}"
        )
    return {
        "sampled_at": _utc_now(),
        "monotonic_seconds": time.monotonic(),
        "physical_gpu": observed_gpu,
        "memory_used_mib": memory_mib,
        "utilization_gpu_percent": utilization_percent,
    }


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise EvidenceError(f"cannot find an existing ancestor for {path}")
        candidate = parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def _host_resource_sample(output_root: Path) -> dict[str, Any]:
    try:
        fields: dict[str, int] = {}
        with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                name, raw = line.split(":", 1)
                value = raw.strip().split()[0]
                fields[name] = int(value)
        available_gib = fields["MemAvailable"] / (1024.0 * 1024.0)
    except (OSError, KeyError, ValueError) as exc:
        raise EvidenceError(f"cannot sample available host memory: {exc}") from exc
    disk_root = _nearest_existing_directory(output_root)
    disk_free_gib = shutil.disk_usage(disk_root).free / float(1024**3)
    return {
        "sampled_at": _utc_now(),
        "available_host_memory_gib": available_gib,
        "output_disk_probe_path": str(disk_root),
        "output_disk_free_gib": disk_free_gib,
    }


def _gpu_is_launch_idle(sample: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    try:
        memory_mib, utilization_percent = _validated_gpu_sample_values(
            sample, "GPU launch sample"
        )
        return (
            memory_mib < int(policy["gpu_memory_threshold_mib"])
            and utilization_percent < int(policy["gpu_utilization_threshold_percent"])
        )
    except (EvidenceError, KeyError, TypeError, ValueError):
        return False


def _acquire_resource_lease(suite: SuiteDefinition, job: SuiteJob) -> ResourceLease | None:
    """Acquire and prove a physical-GPU launch gate for one job."""

    if not _resource_policy_enforced(job):
        return None
    policy = dict(job.resource_policy or {})
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - enforced suites are Linux-only
        raise EvidenceError("runtime resource enforcement requires fcntl/flock") from exc

    preferred = int(policy["preferred_physical_gpu"])
    eligible = [int(value) for value in policy["eligible_physical_gpus"]]
    candidates = [preferred, *(value for value in eligible if value != preferred)]
    interval = int(policy["sample_interval_seconds"])
    deadline = time.monotonic() + float(policy["gate_wait_timeout_hours"]) * 3600.0
    attempt = 0
    while True:
        if time.monotonic() >= deadline:
            raise EvidenceError("resource launch gate timed out without an eligible idle GPU")
        attempt += 1
        host = _host_resource_sample(suite.output_root)
        host_ok = (
            float(host["available_host_memory_gib"])
            >= float(policy["minimum_available_host_memory_gib"])
            and float(host["output_disk_free_gib"])
            >= float(policy["minimum_output_disk_free_gib"])
        )
        first = {gpu: _gpu_sample(gpu) for gpu in candidates}
        time.sleep(interval)
        second = {gpu: _gpu_sample(gpu) for gpu in candidates}
        if not host_ok:
            continue
        for physical_gpu in candidates:
            if not (
                _gpu_is_launch_idle(first[physical_gpu], policy)
                and _gpu_is_launch_idle(second[physical_gpu], policy)
            ):
                continue
            lock_path = Path("/tmp") / f"com_compression_gpu_{physical_gpu}.lock"
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
                continue
            try:
                post_first = _gpu_sample(physical_gpu)
                time.sleep(interval)
                post_second = _gpu_sample(physical_gpu)
                post_host = _host_resource_sample(suite.output_root)
                post_host_ok = (
                    float(post_host["available_host_memory_gib"])
                    >= float(policy["minimum_available_host_memory_gib"])
                    and float(post_host["output_disk_free_gib"])
                    >= float(policy["minimum_output_disk_free_gib"])
                )
                if not (
                    _gpu_is_launch_idle(post_first, policy)
                    and _gpu_is_launch_idle(post_second, policy)
                    and post_host_ok
                ):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                    continue
                evidence = {
                    "schema_version": RESOURCE_GATE_SCHEMA_VERSION,
                    "suite_id": job.suite_id,
                    "job_id": job.job_id,
                    "suite_config_sha256": job.suite_config_sha256,
                    "job_config_sha256": job.job_config_sha256,
                    "execution_fingerprint_sha256": job.execution_fingerprint_sha256,
                    "policy_sha256": _object_sha256(policy),
                    "policy": policy,
                    "attempt": attempt,
                    "selected_physical_gpu": physical_gpu,
                    "cuda_device_order": "PCI_BUS_ID",
                    "cuda_visible_devices": str(physical_gpu),
                    "lock_path": str(lock_path),
                    "lock_acquired": True,
                    "pre_lock_samples": [first[physical_gpu], second[physical_gpu]],
                    "post_lock_samples": [post_first, post_second],
                    "host_pre_lock": host,
                    "host_post_lock": post_host,
                    "gate_passed_at": _utc_now(),
                    "gate_passed": True,
                }
                _write_json_atomic(job.resource_gate_path, evidence)
                return ResourceLease(
                    selected_gpu=physical_gpu,
                    lock_path=lock_path,
                    descriptor=descriptor,
                    gate_evidence=evidence,
                )
            except BaseException:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
                raise


def _child_max_rss_gib() -> float:
    try:
        import resource

        raw = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    except (ImportError, OSError, ValueError) as exc:  # pragma: no cover - Linux production path
        raise EvidenceError(f"cannot read child maximum RSS: {exc}") from exc
    divisor = float(1024**3) if sys.platform == "darwin" else float(1024**2)
    return raw / divisor


def _start_gpu_runtime_monitor(
    physical_gpu: int,
    *,
    interval_seconds: float,
) -> tuple[threading.Event, list[dict[str, Any]], list[str], threading.Thread]:
    stop = threading.Event()
    samples: list[dict[str, Any]] = []
    errors: list[str] = []

    def monitor() -> None:
        while True:
            try:
                samples.append(_gpu_sample(physical_gpu))
            except EvidenceError as exc:
                errors.append(str(exc))
                return
            if stop.wait(interval_seconds):
                return

    thread = threading.Thread(
        target=monitor,
        name=f"gpu-{physical_gpu}-resource-monitor",
        daemon=True,
    )
    thread.start()
    return stop, samples, errors, thread


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_json_sha256(raw: bytes) -> str:
    """Hash declarative JSON independently of Git's CRLF checkout policy."""

    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _decode_json_object(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"expected a JSON object in {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read valid JSON from {path}: {exc}") from exc
    return _decode_json_object(raw, path)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _safe_relative_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise EvidenceError(f"artifact path escapes job directory: {relative!r}") from exc
    return candidate


def _require_nonempty_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise EvidenceError(f"missing or empty expected output: {path}")


def _validate_string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise SuiteConfigError(f"{name} must be a non-empty list of strings")
    if len(value) != len(set(value)):
        raise SuiteConfigError(f"{name} contains duplicate values")
    return list(value)


def _validate_int_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or not value or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise SuiteConfigError(f"{name} must be a non-empty list of integers")
    if len(value) != len(set(value)):
        raise SuiteConfigError(f"{name} contains duplicate values")
    return list(value)


def _validate_rate_list(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise SuiteConfigError(f"{name} must be a non-empty list")
    rates: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SuiteConfigError(f"{name} contains a non-numeric rate")
        rate = float(item)
        if not math.isfinite(rate) or rate <= 0.0 or rate > 1.0:
            raise SuiteConfigError(f"{name} rates must be finite values in (0, 1]")
        rates.append(rate)
    if len(rates) != len(set(rates)):
        raise SuiteConfigError(f"{name} contains duplicate rates")
    return rates


def _validate_safe_protocol_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SuiteConfigError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise SuiteConfigError(f"{name} must be a safe repository-relative path")
    if path.suffix.lower() != ".json":
        raise SuiteConfigError(f"{name} must identify a JSON manifest")
    return path.as_posix()


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SuiteConfigError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_analysis_plan(plan: object, expected_strategies: Sequence[str]) -> dict[str, Any] | None:
    if plan is None:
        return None
    if not isinstance(plan, dict):
        raise SuiteConfigError("analysis_plan must be an object")
    exact_scalars = {
        "schema_version": "confirmatory_hessian_endpoint_screen_analysis.v1",
        "freeze_status": "analysis_config_frozen_before_confirmatory_test_evaluation",
        "primary_metric": "heldout_nll",
        "replicate_unit": "seed",
        "rate_pairing": "the three rates are paired within each seed",
        "fixed_test_window_role": "paired_diagnostic_only_not_independent_replicates",
        "inference": "exact_paired_randomization_on_eight_seed_level_differences",
        "multiplicity": "holm_across_co_primary_family",
        "orthogonality_role": "mechanism_diagnostic_only",
        "within_seed_rate_aggregation": "unweighted_arithmetic_mean_of_candidate_minus_reference_heldout_nll_at_frozen_rates_0p258_0p275_0p300",
        "test_statistic": "mean_of_eight_seed_level_rate_averaged_differences",
        "randomization": "one_sided_exact_sign_flip_all_2_power_8_assignments",
        "zero_difference_handling": "retained_as_zero_under_all_sign_flips",
        "p_value": "count(T_perm <= T_obs)/256",
    }
    for name, expected in exact_scalars.items():
        if plan.get(name) != expected:
            raise SuiteConfigError(f"analysis_plan.{name} must be {expected!r}")
    for name in (
        "data_split_preregistration",
        "rate_match_gate",
        "orthogonality_success_rule",
    ):
        if not isinstance(plan.get(name), str) or not plan[name]:
            raise SuiteConfigError(f"analysis_plan.{name} must be non-empty")
    allowed = set(expected_strategies) | {"dense"}

    def validate_contrast(value: object, name: str) -> tuple[str, str]:
        if not isinstance(value, dict):
            raise SuiteConfigError(f"{name} must be an object")
        candidate = value.get("candidate")
        reference = value.get("reference")
        if candidate not in allowed or reference not in allowed or candidate == reference:
            raise SuiteConfigError(f"{name} must name two distinct declared strategies")
        if value.get("directional_hypothesis") != "candidate_lower":
            raise SuiteConfigError(f"{name}.directional_hypothesis must be candidate_lower")
        if not isinstance(value.get("design_label"), str) or not value["design_label"]:
            raise SuiteConfigError(f"{name}.design_label must be non-empty")
        return str(candidate), str(reference)

    primary = plan.get("primary_contrast")
    primary_pair = validate_contrast(primary, "analysis_plan.primary_contrast")
    if primary_pair != ("Q+S+L_QL_budget", "Q+L"):
        raise SuiteConfigError("analysis_plan primary contrast must be strict same-byte Q+S+L versus Q+L")
    if not isinstance(primary, dict) or primary.get("rate_match_rule") != PRIMARY_RATE_MATCH_RULE:
        raise SuiteConfigError(
            "analysis_plan.primary_contrast.rate_match_rule must require exact actual serialized-byte equality"
        )
    co_primary = plan.get("co_primary_contrasts")
    if not isinstance(co_primary, list) or len(co_primary) != 2:
        raise SuiteConfigError("analysis_plan.co_primary_contrasts must contain two contrasts")
    co_pairs = {
        validate_contrast(value, f"analysis_plan.co_primary_contrasts[{index}]")
        for index, value in enumerate(co_primary)
    }
    if co_pairs != {("Q+S_OBS+L", "Q+S_OBS"), ("Q+S_OBS+L", "Q+L")}:
        raise SuiteConfigError("analysis_plan co-primary OBS contrasts differ from the frozen design")
    if any(
        not isinstance(value, dict)
        or value.get("rate_comparison_rule") != CO_PRIMARY_RATE_COMPARISON_RULE
        for value in co_primary
    ):
        raise SuiteConfigError(
            "analysis_plan co-primary contrasts must be labelled actual-rate/Pareto only"
        )
    for name in ("required_reporting", "orthogonality_required_diagnostics"):
        _validate_string_list(plan.get(name), f"analysis_plan.{name}")
    return plan


def _validate_resource_policy(policy: object) -> dict[str, Any] | None:
    if policy is None:
        return None
    if not isinstance(policy, dict):
        raise SuiteConfigError("resource_policy must be an object")
    if policy.get("enforce_at_runtime") is not True:
        # Older planning-only suites keep their textual policy but do not
        # silently acquire a machine-specific GPU lease.
        return policy
    eligible = policy.get("eligible_physical_gpus")
    if (
        not isinstance(eligible, list)
        or not eligible
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in eligible)
        or len(eligible) != len(set(eligible))
    ):
        raise SuiteConfigError("resource_policy.eligible_physical_gpus is invalid")
    preferred = policy.get("preferred_physical_gpu")
    if isinstance(preferred, bool) or not isinstance(preferred, int) or preferred not in eligible:
        raise SuiteConfigError("resource_policy.preferred_physical_gpu must be eligible")
    positive_integer_fields = (
        "gpu_memory_threshold_mib",
        "gpu_utilization_threshold_percent",
        "sample_interval_seconds",
        "gate_wait_timeout_hours",
        "minimum_available_host_memory_gib",
        "minimum_output_disk_free_gib",
        "sentinel_timeout_hours",
        "three_tensor_pair_timeout_hours",
        "maximum_gpu_memory_mib",
        "maximum_rss_gib",
    )
    for field in positive_integer_fields:
        value = policy.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SuiteConfigError(f"resource_policy.{field} must be a positive integer")
    if int(policy["gpu_utilization_threshold_percent"]) > 100:
        raise SuiteConfigError("resource_policy GPU utilization threshold exceeds 100")
    return policy


def validate_suite_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SuiteConfigError("suite config must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SuiteConfigError(f"schema_version must be {SCHEMA_VERSION!r}")
    suite_id = payload.get("suite_id")
    if not isinstance(suite_id, str) or not SAFE_ID.fullmatch(suite_id):
        raise SuiteConfigError("suite_id must be a safe non-empty identifier")
    for field in ("output_root", "runner"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise SuiteConfigError(f"{field} must be a non-empty string")
    expected_outputs = _validate_string_list(payload.get("expected_outputs"), "expected_outputs")
    if any(Path(item).is_absolute() or ".." in Path(item).parts for item in expected_outputs):
        raise SuiteConfigError("expected_outputs must be safe relative paths")
    expected_strategies = _validate_string_list(payload.get("expected_strategies"), "expected_strategies")
    _validate_analysis_plan(payload.get("analysis_plan"), expected_strategies)
    evidence_contract = payload.get("evidence_contract")
    if not isinstance(evidence_contract, dict):
        raise SuiteConfigError("evidence_contract must be an object")
    protocol_interface = evidence_contract.get("protocol_manifest_interface_supported")
    if not isinstance(protocol_interface, bool):
        raise SuiteConfigError("evidence_contract.protocol_manifest_interface_supported must be boolean")
    declared_common = payload.get("common")
    declared_two_stage = (
        isinstance(declared_common, dict)
        and declared_common.get("two_stage_selection") is True
    )
    if protocol_interface and declared_two_stage:
        raise SuiteConfigError(
            "the immutable confirmatory protocol and split-based two-stage selection "
            "cannot be enabled in the same suite"
        )
    expected_window_policy = (
        PROTOCOL_WINDOW_POLICY
        if protocol_interface
        else (
            TWO_STAGE_WINDOW_POLICY
            if declared_two_stage
            else SHARED_SEQUENTIAL_WINDOW_POLICY
        )
    )
    if evidence_contract.get("current_data_window_policy") != expected_window_policy:
        raise SuiteConfigError(
            "evidence_contract.current_data_window_policy does not match the declared protocol mode"
        )
    if evidence_contract.get("multi_seed_aggregation_requires_consumed_protocol_manifest") is not True:
        raise SuiteConfigError("multi-seed aggregation must require a consumed protocol manifest")
    expected_default_role = "confirmatory" if protocol_interface else "scalability_smoke"
    if evidence_contract.get("default_evidence_role") != expected_default_role:
        raise SuiteConfigError(f"the default evidence role must be {expected_default_role}")
    _validate_resource_policy(payload.get("resource_policy"))

    common = payload.get("common")
    if not isinstance(common, dict):
        raise SuiteConfigError("common must be an object")
    unknown_common = set(common) - (set(RUNNER_ARGUMENT_ORDER) - PER_JOB_ARGUMENTS)
    if unknown_common:
        raise SuiteConfigError(f"unsupported common runner arguments: {sorted(unknown_common)}")
    for name in BOOLEAN_RUNNER_ARGUMENTS.intersection(common):
        if not isinstance(common[name], bool):
            raise SuiteConfigError(f"common.{name} must be boolean")
    for required in (
        "calib_limit",
        "eval_limit",
        "sequence_length",
        "batch_size",
        "texts_per_batch_window",
        "bits",
    ):
        if isinstance(common.get(required), bool) or not isinstance(common.get(required), int) or common[required] <= 0:
            raise SuiteConfigError(f"common.{required} must be a positive integer")
    if not common.get("emit_codec_artifacts") or not common.get("enforce_serialized_rate_cap"):
        raise SuiteConfigError("large-scale physical-rate jobs require artifact emission and serialized caps")
    if (
        not protocol_interface
        and common.get("skip_comfort")
        and common.get("two_stage_selection") is not True
    ) or common.get("proxy_only"):
        raise SuiteConfigError("large-scale endpoint jobs must retain held-out NLL and loss-landscape evidence")
    if common.get("two_stage_selection") is True:
        for field in ("calibration_split", "selection_split", "test_split"):
            if not isinstance(common.get(field), str) or not common[field]:
                raise SuiteConfigError(f"common.{field} must be a non-empty string")
        if len(
            {
                str(common["calibration_split"]),
                str(common["selection_split"]),
                str(common["test_split"]),
            }
        ) != 3:
            raise SuiteConfigError(
                "two-stage selection requires three distinct dataset splits"
            )
        for field in ("selection_limit", "selection_top_k"):
            value = common.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SuiteConfigError(f"common.{field} must be a positive integer")
        if int(common["selection_top_k"]) < 2:
            raise SuiteConfigError("common.selection_top_k must be at least two")
        if common.get("rate_allocation") != "global_exact":
            raise SuiteConfigError(
                "two-stage selection requires common.rate_allocation=global_exact"
            )
        if common.get("include_global_single_component_controls") is not True:
            raise SuiteConfigError(
                "two-stage joint-value suites require global single-component controls"
            )
        required_two_stage_outputs = {
            "allocation_validation_rerank.csv",
            "allocation_validation_window_nll.csv",
            "endpoint_window_nll.csv",
        }
        missing_outputs = required_two_stage_outputs.difference(expected_outputs)
        if missing_outputs:
            raise SuiteConfigError(
                "two-stage suites must retain validation/test evidence outputs: "
                f"{sorted(missing_outputs)}"
            )

    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise SuiteConfigError("stages must be a non-empty list")
    stage_ids: set[str] = set()
    for index, stage in enumerate(stages):
        where = f"stages[{index}]"
        if not isinstance(stage, dict):
            raise SuiteConfigError(f"{where} must be an object")
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not SAFE_ID.fullmatch(stage_id):
            raise SuiteConfigError(f"{where}.id must be a safe identifier")
        if stage_id in stage_ids:
            raise SuiteConfigError(f"duplicate stage id: {stage_id}")
        stage_ids.add(stage_id)
        for field in ("lane", "model", "model_scale"):
            if not isinstance(stage.get(field), str) or not stage[field]:
                raise SuiteConfigError(f"{where}.{field} must be a non-empty string")
        evidence_role = stage.get("evidence_role")
        if evidence_role not in {"scalability_smoke", "confirmatory"}:
            raise SuiteConfigError(f"{where}.evidence_role must be scalability_smoke or confirmatory")
        protocol_consumed = stage.get("protocol_manifest_consumed")
        if not isinstance(protocol_consumed, bool):
            raise SuiteConfigError(f"{where}.protocol_manifest_consumed must be boolean")
        if "protocol_seed" in stage:
            raise SuiteConfigError(f"{where}.protocol_seed is derived from each expanded job seed")
        aggregation_allowed = stage.get("seed_aggregation_allowed")
        if not isinstance(aggregation_allowed, bool):
            raise SuiteConfigError(f"{where}.seed_aggregation_allowed must be boolean")
        independence = stage.get("data_window_independence")
        if not isinstance(independence, str) or not independence:
            raise SuiteConfigError(f"{where}.data_window_independence must be non-empty")
        if protocol_consumed:
            if not protocol_interface:
                raise SuiteConfigError(
                    f"{where} claims a consumed protocol manifest, but the current runner cannot consume one"
                )
            if evidence_role != "confirmatory":
                raise SuiteConfigError(f"{where} must be confirmatory when consuming a protocol manifest")
            if aggregation_allowed is not True:
                raise SuiteConfigError(
                    f"{where} must explicitly allow seed aggregation for the confirmatory protocol"
                )
            if independence != PROTOCOL_WINDOW_INDEPENDENCE:
                raise SuiteConfigError(
                    f"{where}.data_window_independence must exactly describe disjoint calibration and fixed test windows"
                )
            _validate_safe_protocol_path(stage.get("protocol_manifest"), f"{where}.protocol_manifest")
            _validate_sha256(stage.get("protocol_manifest_sha256"), f"{where}.protocol_manifest_sha256")
            if stage.get("protocol_eval_role") != "test":
                raise SuiteConfigError(f"{where}.protocol_eval_role must be test for confirmatory endpoints")
        else:
            unexpected_protocol_fields = {
                name
                for name in (
                    "protocol_manifest",
                    "protocol_manifest_sha256",
                    "protocol_seed",
                    "protocol_eval_role",
                )
                if name in stage
            }
            if unexpected_protocol_fields:
                raise SuiteConfigError(
                    f"{where} declares protocol inputs without protocol_manifest_consumed=true: "
                    f"{sorted(unexpected_protocol_fields)}"
                )
            if evidence_role == "confirmatory":
                raise SuiteConfigError(
                    f"{where} cannot be confirmatory without a runner-consumed protocol manifest"
                )
            if aggregation_allowed:
                raise SuiteConfigError(
                    f"{where} cannot aggregate seeds without a runner-consumed protocol manifest"
                )
            expected_independence = (
                TWO_STAGE_WINDOW_INDEPENDENCE
                if common.get("two_stage_selection") is True
                else SHARED_SEQUENTIAL_INDEPENDENCE
            )
            if independence != expected_independence:
                raise SuiteConfigError(
                    f"{where}.data_window_independence must disclose the active split/window policy"
                )
        if stage.get("model_availability") not in {"required", "optional"}:
            raise SuiteConfigError(f"{where}.model_availability must be required or optional")
        if not isinstance(stage.get("availability_note"), str) or not stage["availability_note"]:
            raise SuiteConfigError(f"{where}.availability_note must be non-empty")
        override = stage.get("model_override_env")
        if override is not None and (
            not isinstance(override, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", override)
        ):
            raise SuiteConfigError(f"{where}.model_override_env is not a valid environment variable")
        if protocol_consumed and override is not None:
            raise SuiteConfigError(
                f"{where}.model_override_env is prohibited for an immutable confirmatory protocol"
            )
        if not isinstance(stage.get("revision", ""), str):
            raise SuiteConfigError(f"{where}.revision must be a string")
        snapshot_fields = (
            "model_snapshot_manifest",
            "model_snapshot_manifest_sha256",
            "model_snapshot_aggregate_sha256",
        )
        snapshot_present = [bool(stage.get(name)) for name in snapshot_fields]
        if any(snapshot_present) != all(snapshot_present):
            raise SuiteConfigError(
                f"{where} must declare model snapshot manifest path, file SHA and aggregate SHA together"
            )
        if all(snapshot_present):
            _validate_safe_protocol_path(
                stage["model_snapshot_manifest"], f"{where}.model_snapshot_manifest"
            )
            _validate_sha256(
                stage["model_snapshot_manifest_sha256"],
                f"{where}.model_snapshot_manifest_sha256",
            )
            _validate_sha256(
                stage["model_snapshot_aggregate_sha256"],
                f"{where}.model_snapshot_aggregate_sha256",
            )
        runner_overrides = stage.get("runner_overrides", {})
        if not isinstance(runner_overrides, dict):
            raise SuiteConfigError(f"{where}.runner_overrides must be an object")
        allowed_stage_overrides = set(RUNNER_ARGUMENT_ORDER) - PER_JOB_ARGUMENTS
        unknown_stage_overrides = set(runner_overrides) - allowed_stage_overrides
        if unknown_stage_overrides:
            raise SuiteConfigError(
                f"{where}.runner_overrides contains unsupported arguments: "
                f"{sorted(unknown_stage_overrides)}"
            )
        for name in BOOLEAN_RUNNER_ARGUMENTS.intersection(runner_overrides):
            if not isinstance(runner_overrides[name], bool):
                raise SuiteConfigError(f"{where}.runner_overrides.{name} must be boolean")
        effective_stage_arguments = dict(common)
        effective_stage_arguments.update(runner_overrides)
        if (
            effective_stage_arguments.get("two_stage_selection") is True
        ) != (common.get("two_stage_selection") is True):
            raise SuiteConfigError(
                f"{where}.runner_overrides cannot change the suite-level two-stage protocol"
            )
        if effective_stage_arguments.get("two_stage_selection") is True:
            effective_splits = [
                effective_stage_arguments.get(field)
                for field in ("calibration_split", "selection_split", "test_split")
            ]
            if any(not isinstance(value, str) or not value for value in effective_splits):
                raise SuiteConfigError(
                    f"{where}.runner_overrides must retain non-empty two-stage splits"
                )
            if len(set(effective_splits)) != 3:
                raise SuiteConfigError(
                    f"{where}.runner_overrides must retain three distinct two-stage splits"
                )
            for field in ("selection_limit", "selection_top_k"):
                value = effective_stage_arguments.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise SuiteConfigError(
                        f"{where}.runner_overrides must retain a positive {field}"
                    )
            if int(effective_stage_arguments["selection_top_k"]) < 2:
                raise SuiteConfigError(
                    f"{where}.runner_overrides must retain selection_top_k >= 2"
                )
            if effective_stage_arguments.get("rate_allocation") != "global_exact":
                raise SuiteConfigError(
                    f"{where}.runner_overrides must retain global_exact allocation"
                )
            if (
                effective_stage_arguments.get(
                    "include_global_single_component_controls"
                )
                is not True
            ):
                raise SuiteConfigError(
                    f"{where}.runner_overrides must retain global no-joint controls"
                )
        if not effective_stage_arguments.get("emit_codec_artifacts") or not effective_stage_arguments.get(
            "enforce_serialized_rate_cap"
        ):
            raise SuiteConfigError(
                f"{where}.runner_overrides must retain artifact emission and serialized caps"
            )
        if (
            (
                not protocol_interface
                and effective_stage_arguments.get("skip_comfort")
                and effective_stage_arguments.get("two_stage_selection") is not True
            )
            or effective_stage_arguments.get("proxy_only")
        ):
            raise SuiteConfigError(
                f"{where}.runner_overrides must retain held-out NLL and loss-landscape evidence"
            )
        _validate_int_list(stage.get("seeds"), f"{where}.seeds")
        _validate_rate_list(stage.get("rates"), f"{where}.rates")
        scope = stage.get("tensor_scope")
        if not isinstance(scope, dict):
            raise SuiteConfigError(f"{where}.tensor_scope must be an object")
        if not isinstance(scope.get("id"), str) or not SAFE_ID.fullmatch(scope["id"]):
            raise SuiteConfigError(f"{where}.tensor_scope.id must be a safe identifier")
        if not isinstance(scope.get("claim_scope"), str) or not scope["claim_scope"]:
            raise SuiteConfigError(f"{where}.tensor_scope.claim_scope must be non-empty")
        modules = _validate_string_list(scope.get("module_types"), f"{where}.tensor_scope.module_types")
        layers = _validate_int_list(scope.get("layers"), f"{where}.tensor_scope.layers")
        if any(layer < 0 for layer in layers):
            raise SuiteConfigError(f"{where}.tensor_scope.layers must be non-negative")
        expected = scope.get("expected_selected_tensors")
        if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
            raise SuiteConfigError(f"{where}.tensor_scope.expected_selected_tensors must be positive")
        if expected != len(modules) * len(layers):
            raise SuiteConfigError(
                f"{where}.tensor_scope.expected_selected_tensors must equal module_types x layers"
            )
        max_modules = scope.get("max_modules")
        if isinstance(max_modules, bool) or not isinstance(max_modules, int) or max_modules < 0:
            raise SuiteConfigError(f"{where}.tensor_scope.max_modules must be a non-negative integer")
        if max_modules and max_modules != expected:
            raise SuiteConfigError(f"{where}.tensor_scope.max_modules must be 0 or the expected tensor count")
        identity_pairs = scope.get("endpoint_identity_pairs", [])
        if not isinstance(identity_pairs, list):
            raise SuiteConfigError(f"{where}.tensor_scope.endpoint_identity_pairs must be a list")
        for pair_index, pair in enumerate(identity_pairs):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or pair[0] == pair[1]
                or any(name not in expected_strategies for name in pair)
            ):
                raise SuiteConfigError(
                    f"{where}.tensor_scope.endpoint_identity_pairs[{pair_index}] is invalid"
                )
    return payload


def _validate_protocol_manifest_contract(raw: Mapping[str, Any], repo_root: Path) -> None:
    protocol_stages = [stage for stage in raw["stages"] if stage["protocol_manifest_consumed"]]
    if not protocol_stages:
        return
    relative_paths = {str(stage["protocol_manifest"]) for stage in protocol_stages}
    expected_digests = {str(stage["protocol_manifest_sha256"]) for stage in protocol_stages}
    if len(relative_paths) != 1 or len(expected_digests) != 1:
        raise SuiteConfigError("one suite must use one immutable protocol manifest")
    relative = next(iter(relative_paths))
    expected_sha = next(iter(expected_digests))
    root = repo_root.resolve()
    manifest_path = (root / relative).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise SuiteConfigError("protocol manifest resolves outside the repository") from exc
    if not manifest_path.is_file():
        raise SuiteConfigError(f"protocol manifest does not exist: {manifest_path}")
    raw_bytes = manifest_path.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise SuiteConfigError(
            f"protocol manifest SHA-256 differs: {actual_sha} != {expected_sha}"
        )
    try:
        manifest = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuiteConfigError(f"protocol manifest is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise SuiteConfigError(f"protocol manifest schema must be {PROTOCOL_SCHEMA_VERSION}")
    if manifest.get("status") != "preregistered_data_split_manifest":
        raise SuiteConfigError("protocol manifest status is not preregistered_data_split_manifest")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list):
        raise SuiteConfigError("protocol manifest seeds is invalid")
    allocation_counts = manifest.get("allocation_counts")
    tokenization = manifest.get("tokenization")
    windows = manifest.get("windows")
    model = manifest.get("model")
    if (
        not isinstance(allocation_counts, dict)
        or not isinstance(tokenization, dict)
        or not isinstance(windows, dict)
        or not isinstance(model, dict)
    ):
        raise SuiteConfigError("protocol manifest lacks allocation/tokenization/model/window records")
    model_id = model.get("model_id")
    snapshot_commit = model.get("snapshot_commit")
    if not isinstance(model_id, str) or not model_id:
        raise SuiteConfigError("protocol manifest model.model_id is invalid")
    if not isinstance(snapshot_commit, str) or not snapshot_commit:
        raise SuiteConfigError("protocol manifest model.snapshot_commit is invalid")
    if tokenization.get("snapshot_commit") != snapshot_commit:
        raise SuiteConfigError("protocol tokenizer snapshot differs from the frozen model snapshot")
    if tokenization.get("tokenizer_class") != "GPTNeoXTokenizerFast":
        raise SuiteConfigError("protocol tokenizer class differs from GPTNeoXTokenizerFast")
    common = raw["common"]
    if tokenization.get("window_token_length") != common["sequence_length"]:
        raise SuiteConfigError("protocol token length differs from common.sequence_length")
    if allocation_counts.get("calibration_windows_per_seed") != common["calib_limit"]:
        raise SuiteConfigError("protocol calibration-window count differs from common.calib_limit")
    for stage in protocol_stages:
        if stage.get("model") != model_id:
            raise SuiteConfigError(
                f"{stage['id']}: stage model differs from the protocol manifest model"
            )
        if stage.get("revision") != snapshot_commit:
            raise SuiteConfigError(
                f"{stage['id']}: stage revision differs from the protocol manifest snapshot"
            )
        stage_seeds = list(stage["seeds"])
        if any(seed not in seeds for seed in stage_seeds):
            raise SuiteConfigError(f"{stage['id']}: seed is absent from the protocol manifest")
        role = str(stage["protocol_eval_role"])
        role_windows = windows.get(role)
        if not isinstance(role_windows, list) or len(role_windows) != common["eval_limit"]:
            raise SuiteConfigError(
                f"{stage['id']}: protocol {role} window count differs from common.eval_limit"
            )
        calibration = windows.get("calibration_by_seed")
        if not isinstance(calibration, dict):
            raise SuiteConfigError("protocol manifest calibration_by_seed is invalid")
        for seed in stage_seeds:
            selected = calibration.get(str(seed))
            if not isinstance(selected, list) or len(selected) != common["calib_limit"]:
                raise SuiteConfigError(f"{stage['id']}: protocol calibration count differs for seed {seed}")


def _validate_model_snapshot_contracts(raw: Mapping[str, Any], repo_root: Path) -> None:
    root = repo_root.resolve()
    checked: set[str] = set()
    for index, stage in enumerate(raw["stages"]):
        relative = stage.get("model_snapshot_manifest")
        if not relative or str(relative) in checked:
            continue
        checked.add(str(relative))
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SuiteConfigError(
                f"stages[{index}].model_snapshot_manifest escapes the repository"
            ) from exc
        if not path.is_file():
            raise SuiteConfigError(f"missing model snapshot manifest: {path}")
        raw_bytes = path.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != stage["model_snapshot_manifest_sha256"]:
            raise SuiteConfigError(f"model snapshot manifest SHA-256 differs: {path}")
        try:
            manifest = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SuiteConfigError(f"invalid model snapshot manifest JSON: {path}") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != "model_snapshot_manifest.v1"
            or manifest.get("aggregate_sha256")
            != stage["model_snapshot_aggregate_sha256"]
        ):
            raise SuiteConfigError(f"model snapshot aggregate contract differs: {path}")


def load_suite_definition(
    config_path: Path,
    *,
    repo_root: Path | None = None,
    output_root_override: Path | None = None,
    runner_override: Path | None = None,
) -> SuiteDefinition:
    config_path = Path(config_path).resolve()
    try:
        raw_bytes = config_path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuiteConfigError(f"cannot load suite config {config_path}: {exc}") from exc
    raw = validate_suite_payload(payload)
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    _validate_protocol_manifest_contract(raw, root)
    _validate_model_snapshot_contracts(raw, root)
    output_root = (
        Path(output_root_override).resolve()
        if output_root_override is not None
        else (root / str(raw["output_root"])).resolve()
    )
    runner = (
        Path(runner_override).resolve()
        if runner_override is not None
        else (root / str(raw["runner"])).resolve()
    )
    if not runner.is_file():
        raise SuiteConfigError(f"numerical runner does not exist: {runner}")
    return SuiteDefinition(
        config_path=config_path,
        repo_root=root,
        raw=raw,
        config_sha256=_portable_json_sha256(raw_bytes),
        output_root=output_root,
        runner=runner,
        expected_outputs=tuple(raw["expected_outputs"]),
        expected_strategies=tuple(raw["expected_strategies"]),
        common=dict(raw["common"]),
        stages=tuple(dict(stage) for stage in raw["stages"]),
    )


def collect_numerical_source_snapshot(suite: SuiteDefinition) -> dict[str, dict[str, Any]]:
    protocol_stages = [stage for stage in suite.stages if stage["protocol_manifest_consumed"]]
    if protocol_stages:
        protocol_paths = {str(stage["protocol_manifest"]) for stage in protocol_stages}
        if len(protocol_paths) != 1:
            raise SuiteConfigError("one suite must use one immutable protocol manifest")
        candidates = {
            "runner": suite.runner,
            "protocol_consumer": suite.repo_root / "scripts" / "confirmatory_protocol_windows.py",
            "legacy_runner": suite.repo_root / "scripts" / "run_pretrained_hessian_repair.py",
            "codec": suite.repo_root / "src" / "llm_spectral_dynamics" / "structured" / "codec_artifact.py",
            "hessian_repair": suite.repo_root / "src" / "llm_spectral_dynamics" / "structured" / "hessian_repair.py",
            "base_runner": suite.repo_root / "scripts" / "run_pretrained_llm_orthogonality.py",
            "model_data": suite.repo_root / "src" / "llm_spectral_dynamics" / "structured" / "data.py",
            "protocol_manifest": suite.repo_root / next(iter(protocol_paths)),
        }
        required = set(candidates)
    else:
        # Keep this exact four-file legacy closure stable: the signed pilot job
        # hashes depend on it.
        candidates = {
            "runner": suite.runner,
            "codec": suite.repo_root / "src" / "llm_spectral_dynamics" / "structured" / "codec_artifact.py",
            "hessian_repair": suite.repo_root / "src" / "llm_spectral_dynamics" / "structured" / "hessian_repair.py",
            "base_runner": suite.repo_root / "scripts" / "run_pretrained_llm_orthogonality.py",
        }
        required = {"runner"}
    snapshot_manifests = sorted(
        {
            str(stage["model_snapshot_manifest"])
            for stage in suite.stages
            if stage.get("model_snapshot_manifest")
        }
    )
    if snapshot_manifests:
        candidates["model_snapshot_tool"] = (
            suite.repo_root / "scripts" / "build_model_snapshot_manifest.py"
        )
        required.add("model_snapshot_tool")
    snapshot: dict[str, dict[str, Any]] = {}
    for name, path in candidates.items():
        if not path.is_file():
            if name in required:
                raise SuiteConfigError(f"missing numerical source: {path}")
            continue
        raw = path.read_bytes()
        try:
            relative = path.resolve().relative_to(suite.repo_root).as_posix()
        except ValueError:
            relative = str(path.resolve())
        snapshot[name] = {
            "path": relative,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return snapshot


def load_recorded_numerical_source_snapshot(
    suite: SuiteDefinition,
) -> dict[str, dict[str, Any]] | None:
    """Load an immutable completed suite's historical source closure for audit.

    ``--check`` validates artifacts against the source hashes recorded when the
    jobs ran, not against whichever numerical files happen to be in the current
    worktree.  Execution never calls this path, so a new run cannot masquerade
    as the historical source snapshot.
    """

    manifest_path = suite.output_root / "suite_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SuiteConfigError("recorded suite manifest schema differs")
    if manifest.get("suite_id") != suite.suite_id:
        raise SuiteConfigError("recorded suite manifest id differs")
    if manifest.get("suite_config_sha256") != suite.config_sha256:
        raise SuiteConfigError("recorded suite manifest config hash differs")
    raw_snapshot = manifest.get("numerical_source_snapshot")
    if not isinstance(raw_snapshot, dict) or not raw_snapshot:
        raise SuiteConfigError("recorded suite lacks a numerical source snapshot")
    snapshot: dict[str, dict[str, Any]] = {}
    for name, raw in raw_snapshot.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise SuiteConfigError("recorded numerical source snapshot is invalid")
        path = raw.get("path")
        size = raw.get("size_bytes")
        sha256 = raw.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sha256)
        ):
            raise SuiteConfigError(f"recorded numerical source entry is invalid: {name}")
        snapshot[name] = {"path": path, "size_bytes": size, "sha256": sha256}
    source_sha = _object_sha256(snapshot)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise SuiteConfigError("recorded suite manifest jobs are invalid")
    recorded_hashes = {
        str(row.get("numerical_source_sha256", ""))
        for row in jobs
        if isinstance(row, dict)
    }
    if recorded_hashes != {source_sha}:
        raise SuiteConfigError("recorded job/source hashes are inconsistent")
    return snapshot


def expand_jobs(
    suite: SuiteDefinition,
    *,
    environment: Mapping[str, str] | None = None,
    recorded_model_arguments: Mapping[str, str] | None = None,
) -> list[SuiteJob]:
    env = os.environ if environment is None else environment
    recorded_snapshot = (
        load_recorded_numerical_source_snapshot(suite)
        if recorded_model_arguments is not None
        else None
    )
    source_snapshot = recorded_snapshot or collect_numerical_source_snapshot(suite)
    source_sha = _object_sha256(source_snapshot)
    jobs: list[SuiteJob] = []
    for stage in suite.stages:
        override_name = stage.get("model_override_env")
        override_value = env.get(str(override_name), "").strip() if override_name else ""
        recorded_value = ""
        if recorded_model_arguments is not None:
            recorded_value = str(recorded_model_arguments.get(str(stage["id"]), "")).strip()
        model_argument = recorded_value or override_value or str(stage["model"])
        scope = dict(stage["tensor_scope"])
        for seed in stage["seeds"]:
            for raw_rate in stage["rates"]:
                rate = float(raw_rate)
                effective = dict(suite.common)
                effective.update(dict(stage.get("runner_overrides", {})))
                effective.update(
                    {
                        "model": model_argument,
                        "revision": str(stage.get("revision", "")),
                        "module_types": list(scope["module_types"]),
                        "layers": list(scope["layers"]),
                        "max_modules": int(scope["max_modules"]),
                        "target_ratios": [rate],
                        "endpoint_target": rate,
                        "seed": int(seed),
                    }
                )
                snapshot_manifest = stage.get("model_snapshot_manifest")
                if snapshot_manifest:
                    effective.update(
                        {
                            "model_snapshot_manifest": str(snapshot_manifest),
                            "model_snapshot_manifest_sha256": str(
                                stage["model_snapshot_manifest_sha256"]
                            ),
                            "model_snapshot_aggregate_sha256": str(
                                stage["model_snapshot_aggregate_sha256"]
                            ),
                        }
                    )
                protocol_manifest: str | None = None
                protocol_manifest_sha256: str | None = None
                protocol_seed: int | None = None
                protocol_eval_role: str | None = None
                if stage["protocol_manifest_consumed"]:
                    protocol_manifest = str(stage["protocol_manifest"])
                    protocol_manifest_sha256 = str(stage["protocol_manifest_sha256"])
                    protocol_seed = int(seed)
                    protocol_eval_role = str(stage["protocol_eval_role"])
                    effective.update(
                        {
                            "protocol_manifest": protocol_manifest,
                            "protocol_manifest_sha256": protocol_manifest_sha256,
                            "protocol_seed": protocol_seed,
                            "protocol_eval_role": protocol_eval_role,
                        }
                    )
                provisional = {
                    "suite_id": suite.suite_id,
                    "stage_id": stage["id"],
                    "lane": stage["lane"],
                    "evidence_role": stage["evidence_role"],
                    "protocol_manifest_consumed": stage["protocol_manifest_consumed"],
                    "seed_aggregation_allowed": stage["seed_aggregation_allowed"],
                    "data_window_independence": stage["data_window_independence"],
                    "model_declared": stage["model"],
                    "model_argument": model_argument,
                    "model_scale": stage["model_scale"],
                    "model_availability": stage["model_availability"],
                    "availability_note": stage["availability_note"],
                    "revision": str(stage.get("revision", "")),
                    "seed": int(seed),
                    "target_rate": rate,
                    "tensor_scope": scope,
                    "effective_arguments": effective,
                }
                if (suite.raw.get("resource_policy") or {}).get("enforce_at_runtime") is True:
                    provisional["resource_policy"] = suite.raw["resource_policy"]
                if protocol_manifest is not None:
                    provisional["protocol_manifest"] = protocol_manifest
                    provisional["protocol_manifest_sha256"] = protocol_manifest_sha256
                    provisional["protocol_seed"] = protocol_seed
                    provisional["protocol_eval_role"] = protocol_eval_role
                job_hash = _object_sha256(provisional)
                fingerprint = _object_sha256(
                    {
                        "suite_config_sha256": suite.config_sha256,
                        "job_config_sha256": job_hash,
                        "numerical_source_sha256": source_sha,
                    }
                )
                rate_token = f"{rate:.3f}".replace(".", "p")
                job_id = f"{stage['id']}__seed{seed}__rate{rate_token}"
                jobs.append(
                    SuiteJob(
                        suite_id=suite.suite_id,
                        stage_id=str(stage["id"]),
                        lane=str(stage["lane"]),
                        evidence_role=str(stage["evidence_role"]),
                        protocol_manifest_consumed=bool(stage["protocol_manifest_consumed"]),
                        seed_aggregation_allowed=bool(stage["seed_aggregation_allowed"]),
                        data_window_independence=str(stage["data_window_independence"]),
                        protocol_manifest=protocol_manifest,
                        protocol_manifest_sha256=protocol_manifest_sha256,
                        protocol_seed=protocol_seed,
                        protocol_eval_role=protocol_eval_role,
                        model_declared=str(stage["model"]),
                        model_argument=model_argument,
                        model_scale=str(stage["model_scale"]),
                        model_availability=str(stage["model_availability"]),
                        availability_note=str(stage["availability_note"]),
                        model_override_env=str(override_name) if override_name else None,
                        revision=str(stage.get("revision", "")),
                        seed=int(seed),
                        target_rate=rate,
                        tensor_scope=scope,
                        effective_arguments=effective,
                        output_dir=suite.output_root / "jobs" / job_id,
                        expected_outputs=suite.expected_outputs,
                        expected_strategies=suite.expected_strategies,
                        suite_config_sha256=suite.config_sha256,
                        job_config_sha256=job_hash,
                        numerical_source_snapshot=source_snapshot,
                        numerical_source_sha256=source_sha,
                        execution_fingerprint_sha256=fingerprint,
                        resource_policy=(
                            dict(suite.raw["resource_policy"])
                            if isinstance(suite.raw.get("resource_policy"), dict)
                            else None
                        ),
                        repo_root=suite.repo_root,
                    )
                )
    ids = [job.job_id for job in jobs]
    if len(ids) != len(set(ids)):
        raise SuiteConfigError("expanded job ids are not unique")
    return jobs


def load_recorded_model_arguments(suite: SuiteDefinition) -> dict[str, str]:
    """Load persisted model locators for a portable evidence audit.

    Model overrides are execution-time locators and may be absolute cache paths
    on the machine that produced the artifacts.  A check must recompute every
    job hash from the recorded locator rather than from the current machine's
    environment, while still validating the suite/config identity before the
    locator is trusted.
    """

    manifest_path = suite.output_root / "suite_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise EvidenceError("persisted manifest schema differs")
    if manifest.get("suite_id") != suite.suite_id:
        raise EvidenceError("persisted manifest suite id differs")
    if manifest.get("suite_config_sha256") != suite.config_sha256:
        raise EvidenceError("persisted manifest config hash differs")
    entries = manifest.get("jobs")
    if not isinstance(entries, list):
        raise EvidenceError("persisted manifest jobs is invalid")
    known_stages = {str(stage["id"]) for stage in suite.stages}
    recorded: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise EvidenceError("persisted manifest contains a non-object job")
        stage_id = entry.get("stage_id")
        model_argument = entry.get("model_argument")
        if not isinstance(stage_id, str) or stage_id not in known_stages:
            raise EvidenceError(f"persisted manifest contains unknown stage: {stage_id!r}")
        if not isinstance(model_argument, str) or not model_argument.strip():
            raise EvidenceError(f"{stage_id}: persisted model argument is missing")
        previous = recorded.setdefault(stage_id, model_argument.strip())
        if previous != model_argument.strip():
            raise EvidenceError(f"{stage_id}: persisted model argument differs across jobs")
    return recorded


def build_runner_command(job: SuiteJob, *, python_executable: str, runner: Path) -> list[str]:
    arguments = dict(job.effective_arguments)
    arguments["output_dir"] = str(job.output_dir)
    if (job.resource_policy or {}).get("enforce_at_runtime") is True:
        arguments["resource_gate_manifest"] = str(job.resource_gate_path)
    command = [str(python_executable), str(runner)]
    unknown = set(arguments) - set(RUNNER_ARGUMENT_ORDER)
    if unknown:
        raise SuiteConfigError(f"unsupported effective runner arguments: {sorted(unknown)}")
    for name in RUNNER_ARGUMENT_ORDER:
        if name not in arguments:
            continue
        value = arguments[name]
        flag = "--" + name.replace("_", "-")
        if name in BOOLEAN_RUNNER_ARGUMENTS:
            if value is True:
                command.append(flag)
            elif value is not False:
                raise SuiteConfigError(f"{name} must be boolean")
            continue
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple)):
            serialized = ",".join(str(item) for item in value)
        else:
            serialized = str(value)
        command.extend([flag, serialized])
    return command


def _value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _value_matches(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _layer_index(name: str) -> int | None:
    for pattern in LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def _validate_artifact_file(job_dir: Path, entry: Mapping[str, Any], path_key: str, bytes_key: str) -> dict[str, Any]:
    relative = entry.get(path_key)
    expected_bytes = entry.get(bytes_key)
    expected_sha = entry.get("sha256") if path_key == "path" else entry.get("artifact_sha256")
    if not isinstance(relative, str) or not relative:
        raise EvidenceError(f"artifact manifest has no {path_key}")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise EvidenceError(f"artifact manifest has invalid {bytes_key} for {relative}")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise EvidenceError(f"artifact manifest has invalid SHA-256 for {relative}")
    path = _safe_relative_file(job_dir, relative)
    _require_nonempty_file(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise EvidenceError(f"artifact size mismatch for {path}: {actual_bytes} != {expected_bytes}")
    actual_sha = _file_sha256(path)
    if actual_sha != expected_sha:
        raise EvidenceError(f"artifact SHA-256 mismatch for {path}")
    return {"path": relative, "file_bytes": actual_bytes, "sha256": actual_sha}


def _validate_strategy_csv(job: SuiteJob) -> None:
    path = job.output_dir / "strategy_endpoints.csv"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise EvidenceError(f"cannot read {path}: {exc}") from exc
    observed = {
        str(row.get("strategy"))
        for row in rows
        if math.isclose(float(row.get("target_ratio", "nan")), job.target_rate, rel_tol=0.0, abs_tol=1e-12)
    }
    expected = set(job.expected_strategies)
    if observed != expected:
        raise EvidenceError(
            f"strategy_endpoints.csv methods differ from the declared method set: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )


def _finite_record_number(record: Mapping[str, Any], field: str, context: str) -> float:
    value = record.get(field)
    if isinstance(value, bool):
        raise EvidenceError(f"{context} has invalid {field}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{context} has invalid {field}") from exc
    if not math.isfinite(parsed):
        raise EvidenceError(f"{context} has non-finite {field}")
    return parsed


def _positive_record_integer(record: Mapping[str, Any], field: str, context: str) -> int:
    parsed = _finite_record_number(record, field, context)
    if parsed <= 0.0 or not parsed.is_integer():
        raise EvidenceError(f"{context} has non-positive or non-integral {field}")
    return int(parsed)


def _nonnegative_record_integer(
    record: Mapping[str, Any], field: str, context: str
) -> int:
    parsed = _finite_record_number(record, field, context)
    if parsed < 0.0 or not parsed.is_integer():
        raise EvidenceError(f"{context} has negative or non-integral {field}")
    return int(parsed)


def _validated_gpu_sample_values(
    sample: Mapping[str, Any], context: str
) -> tuple[int, int]:
    memory = _finite_record_number(sample, "memory_used_mib", context)
    utilization = _finite_record_number(sample, "utilization_gpu_percent", context)
    if memory < 0.0 or not memory.is_integer():
        raise EvidenceError(f"{context} has invalid GPU memory")
    if utilization < 0.0 or utilization > 100.0 or not utilization.is_integer():
        raise EvidenceError(f"{context} has invalid GPU utilization")
    return int(memory), int(utilization)


def _read_strategy_table(job: SuiteJob, relative: str) -> dict[str, dict[str, str]]:
    path = job.output_dir / relative
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise EvidenceError(f"cannot read {path}: {exc}") from exc
    names = [str(row.get("strategy", "")) for row in rows]
    if names != list(job.expected_strategies):
        raise EvidenceError(
            f"{relative} strategy order/set differs from the suite contract: "
            f"{names} != {list(job.expected_strategies)}"
        )
    return {name: row for name, row in zip(names, rows)}


def _validate_physical_rates(
    job: SuiteJob,
    manifest_entries: Sequence[Mapping[str, Any]],
    reference_evidence: Mapping[str, Any],
    strategy_evidence: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Recompute physical rates from files and bind all redundant tables to them.

    The legacy flag name ``enforce_serialized_rate_cap`` does not imply that
    every native method is padded to an identical byte count.  All methods are
    therefore audited at their actual serialized rates, while the two
    ``QL_budget`` variants are additionally required to equal ``Q+L`` exactly.
    """

    if "artifact_payloads.csv" not in job.expected_outputs:
        raise EvidenceError("physical-rate jobs must declare artifact_payloads.csv as an expected output")
    endpoint_rows = _read_strategy_table(job, "strategy_endpoints.csv")
    payload_rows = _read_strategy_table(job, "artifact_payloads.csv")
    manifest_by_strategy = {
        str(entry.get("strategy", "")): entry for entry in manifest_entries
    }
    reference_bytes = int(reference_evidence["file_bytes"])
    tolerance = float(job.effective_arguments.get("rate_tolerance", 0.0))
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise EvidenceError("suite job has an invalid rate_tolerance")
    audited: dict[str, dict[str, Any]] = {}

    for strategy in job.expected_strategies:
        artifact = dict(strategy_evidence[strategy])
        artifact_bytes = int(artifact["file_bytes"])
        natural_bytes: int | None = None
        tail_padding_bytes: int | None = None
        actual_ratio = artifact_bytes / reference_bytes
        actual_compression = reference_bytes / artifact_bytes
        if actual_ratio > job.target_rate + tolerance + 1e-12:
            raise EvidenceError(
                f"{strategy} actual serialized ratio {actual_ratio:.12g} exceeds "
                f"target+tolerance {job.target_rate + tolerance:.12g}"
            )

        records: tuple[tuple[str, Mapping[str, Any]], ...] = (
            ("artifact_manifest.json", manifest_by_strategy[strategy]),
            ("artifact_payloads.csv", payload_rows[strategy]),
            ("strategy_endpoints.csv", endpoint_rows[strategy]),
        )
        for source, record in records:
            context = f"{source} {strategy}"
            if not math.isclose(
                _finite_record_number(record, "target_ratio", context),
                job.target_rate,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise EvidenceError(f"{context} target_ratio differs from the job")
            if _positive_record_integer(record, "artifact_file_bytes", context) != artifact_bytes:
                raise EvidenceError(f"{context} artifact_file_bytes differs from the actual file")
            if (
                _positive_record_integer(record, "reference_artifact_file_bytes", context)
                != reference_bytes
            ):
                raise EvidenceError(
                    f"{context} reference_artifact_file_bytes differs from the actual file"
                )
            if not math.isclose(
                _finite_record_number(record, "artifact_to_reference_file_ratio", context),
                actual_ratio,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise EvidenceError(f"{context} serialized artifact ratio is inconsistent")
            if not math.isclose(
                _finite_record_number(record, "artifact_physical_compression_ratio", context),
                actual_compression,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise EvidenceError(f"{context} physical compression ratio is inconsistent")
            if record.get("artifact_path") != artifact["path"]:
                raise EvidenceError(f"{context} artifact_path differs from the actual artifact")
            if record.get("artifact_sha256") != artifact["sha256"]:
                raise EvidenceError(f"{context} artifact SHA-256 differs from the actual artifact")
            if job.effective_arguments.get("rate_allocation") == "global_exact":
                observed_natural = _positive_record_integer(
                    record, "artifact_natural_file_bytes", context
                )
                if observed_natural > artifact_bytes:
                    raise EvidenceError(
                        f"{context} natural artifact bytes exceed the physical file"
                    )
                if natural_bytes is None:
                    natural_bytes = observed_natural
                elif observed_natural != natural_bytes:
                    raise EvidenceError(
                        f"{context} natural artifact bytes differ across evidence tables"
                    )
                observed_tail_padding = _nonnegative_record_integer(
                    record, "artifact_tail_padding_bytes", context
                )
                if observed_natural + observed_tail_padding != artifact_bytes:
                    raise EvidenceError(
                        f"{context} natural bytes plus tail padding do not equal the actual file"
                    )
                if tail_padding_bytes is None:
                    tail_padding_bytes = observed_tail_padding
                elif observed_tail_padding != tail_padding_bytes:
                    raise EvidenceError(
                        f"{context} tail padding differs across evidence tables"
                    )

        artifact.update(
            {
                "reference_file_bytes": reference_bytes,
                "artifact_to_reference_file_ratio": actual_ratio,
                "artifact_physical_compression_ratio": actual_compression,
                "within_target_rate_tolerance": True,
            }
        )
        if natural_bytes is not None:
            artifact["natural_file_bytes"] = natural_bytes
            artifact["tail_padding_bytes"] = int(tail_padding_bytes or 0)
            if strategy == "Q+L" and natural_bytes != artifact_bytes:
                raise EvidenceError(
                    "Q+L is an uncapped reference strategy but its natural bytes "
                    "differ from the actual file"
                )
        audited[strategy] = dict(artifact)

    ql = audited.get("Q+L")
    if ql is not None:
        for strategy in (
            "Q+S+L_QL_budget",
            "Q+S+L_QL_budget_component_scale",
            "Q+S_OBS_global",
            "Q+L_global",
            "Q+S_OBS_or_L_global",
        ):
            if strategy in audited and audited[strategy]["file_bytes"] != ql["file_bytes"]:
                raise EvidenceError(
                    f"{strategy} is not exact-byte matched to Q+L: "
                    f"{audited[strategy]['file_bytes']} != {ql['file_bytes']}"
                )
    return audited


def _validate_finite_endpoint_metrics(job: SuiteJob) -> None:
    rows = _read_strategy_table(job, "strategy_endpoints.csv")
    required = (
        "activation_reconstruction_error",
        "heldout_nll",
        "heldout_perplexity",
        "hessian_cost",
        "normalized_hessian_cost",
        "nll_delta",
        "paired_window_nll_delta_mean",
        "payload_ratio",
        "perplexity_delta",
        "token_risk_p95",
        "worst_token_risk",
    )
    for strategy, row in rows.items():
        for field in required:
            _finite_record_number(row, field, f"strategy_endpoints.csv {strategy}")


def _validate_endpoint_identities(
    job: SuiteJob,
    endpoint_rows: Mapping[str, Mapping[str, Any]],
    audited_artifacts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs = job.tensor_scope.get("endpoint_identity_pairs", [])
    evidence: list[dict[str, Any]] = []
    required_metric_fields = (
        "activation_reconstruction_error",
        "heldout_nll",
        "heldout_perplexity",
        "hessian_cost",
        "normalized_hessian_cost",
        "nll_delta",
        "paired_window_nll_delta_mean",
        "payload_ratio",
        "perplexity_delta",
        "token_risk_p95",
        "worst_token_risk",
    )
    for left, right in pairs:
        left_artifact = audited_artifacts[left]
        right_artifact = audited_artifacts[right]
        for field in ("sha256", "file_bytes", "natural_file_bytes"):
            if left_artifact.get(field) != right_artifact.get(field):
                raise EvidenceError(
                    f"endpoint identity {left}/{right} differs in artifact {field}"
                )
        left_row = endpoint_rows[left]
        right_row = endpoint_rows[right]
        for field in required_metric_fields:
            left_value = _finite_record_number(
                left_row, field, f"strategy_endpoints.csv {left}"
            )
            right_value = _finite_record_number(
                right_row, field, f"strategy_endpoints.csv {right}"
            )
            tolerance = 1e-12 * max(1.0, abs(left_value), abs(right_value))
            if not math.isclose(
                left_value, right_value, rel_tol=1e-12, abs_tol=tolerance
            ):
                raise EvidenceError(
                    f"endpoint identity {left}/{right} differs in {field}"
                )
        evidence.append(
            {
                "left": left,
                "right": right,
                "artifact_sha256": left_artifact["sha256"],
                "artifact_file_bytes": int(left_artifact["file_bytes"]),
                "artifact_natural_file_bytes": int(
                    left_artifact["natural_file_bytes"]
                ),
                "endpoint_metrics_identical": True,
            }
        )
    return evidence


def _job_repo_root(job: SuiteJob) -> Path:
    runner_source = job.numerical_source_snapshot.get("runner")
    if not isinstance(runner_source, dict):
        raise EvidenceError("job numerical source snapshot has no runner")
    raw_path = runner_source.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise EvidenceError("job runner source path is invalid")
    try:
        root = job.repo_root.expanduser().resolve(strict=True)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError(f"job runner source path cannot be resolved: {exc}") from exc
    if path.parent.name != "scripts" or path.parents[1] != root:
        raise EvidenceError("job runner source is not inside the repository scripts directory")
    return root


def _validate_model_snapshot_evidence(
    job: SuiteJob,
    run_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    expected_manifest = job.effective_arguments.get("model_snapshot_manifest")
    evidence = run_config.get("model_snapshot")
    if not expected_manifest:
        if evidence is not None:
            raise EvidenceError("run_config contains undeclared model snapshot evidence")
        return None
    if not isinstance(evidence, dict):
        raise EvidenceError("run_config has no model snapshot verification evidence")
    repo_root = _job_repo_root(job)
    manifest_path = Path(str(expected_manifest)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    try:
        manifest_path = manifest_path.resolve(strict=True)
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read the declared model snapshot manifest: {exc}") from exc
    actual_manifest_sha = hashlib.sha256(raw).hexdigest()
    expected_manifest_sha = str(
        job.effective_arguments["model_snapshot_manifest_sha256"]
    )
    expected_aggregate = str(
        job.effective_arguments["model_snapshot_aggregate_sha256"]
    )
    if actual_manifest_sha != expected_manifest_sha:
        raise EvidenceError("model snapshot manifest file SHA-256 differs at audit time")
    manifest = _decode_json_object(raw, manifest_path)
    if manifest.get("aggregate_sha256") != expected_aggregate:
        raise EvidenceError("model snapshot manifest aggregate differs at audit time")
    try:
        model_dir = Path(job.model_argument).expanduser().resolve(strict=True)
        recorded_model_dir = Path(str(manifest["model_dir"])).expanduser().resolve(
            strict=True
        )
        evidence_manifest_path = Path(str(evidence["manifest_path"])).expanduser().resolve(
            strict=True
        )
        evidence_model_dir = Path(str(evidence["model_dir"])).expanduser().resolve(
            strict=True
        )
    except (KeyError, OSError) as exc:
        raise EvidenceError(f"model snapshot path evidence is invalid: {exc}") from exc
    if model_dir != recorded_model_dir or model_dir != evidence_model_dir:
        raise EvidenceError("model snapshot evidence is not bound to the loaded model directory")
    if evidence_manifest_path != manifest_path:
        raise EvidenceError("model snapshot evidence manifest path differs")
    expected_fields = {
        "schema_version": "model_snapshot_manifest.v1",
        "manifest_sha256": expected_manifest_sha,
        "aggregate_sha256": expected_aggregate,
        "verified_current_tree": True,
    }
    for field, expected in expected_fields.items():
        if evidence.get(field) != expected:
            raise EvidenceError(
                f"model snapshot run evidence {field} differs: {evidence.get(field)!r}"
            )
    file_count = _positive_record_integer(evidence, "file_count", "model_snapshot")
    total_bytes = _positive_record_integer(evidence, "total_bytes", "model_snapshot")
    if file_count != manifest.get("file_count") or total_bytes != manifest.get("total_bytes"):
        raise EvidenceError("model snapshot run evidence size ledger differs from the manifest")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha,
        "aggregate_sha256": expected_aggregate,
        "model_dir": str(model_dir),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "verified_current_tree_before_model_load": True,
    }


def _validate_resource_evidence(
    job: SuiteJob,
    run_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not _resource_policy_enforced(job):
        return None
    policy = dict(job.resource_policy or {})
    try:
        gate_path = job.resource_gate_path.resolve(strict=True)
        gate_raw = gate_path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read the resource gate manifest: {exc}") from exc
    gate_sha = hashlib.sha256(gate_raw).hexdigest()
    gate = _decode_json_object(gate_raw, gate_path)
    required_gate = {
        "schema_version": RESOURCE_GATE_SCHEMA_VERSION,
        "suite_id": job.suite_id,
        "job_id": job.job_id,
        "suite_config_sha256": job.suite_config_sha256,
        "job_config_sha256": job.job_config_sha256,
        "execution_fingerprint_sha256": job.execution_fingerprint_sha256,
        "policy_sha256": _object_sha256(policy),
        "policy": policy,
        "cuda_device_order": "PCI_BUS_ID",
        "lock_acquired": True,
        "gate_passed": True,
    }
    for field, expected in required_gate.items():
        if gate.get(field) != expected:
            raise EvidenceError(f"resource gate {field} differs from the job contract")
    selected = gate.get("selected_physical_gpu")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected not in policy["eligible_physical_gpus"]
    ):
        raise EvidenceError("resource gate selected physical GPU is invalid")
    if gate.get("cuda_visible_devices") != str(selected):
        raise EvidenceError("resource gate CUDA_VISIBLE_DEVICES differs from the physical GPU")
    expected_lock_path = Path("/tmp") / f"com_compression_gpu_{selected}.lock"
    if gate.get("lock_path") != str(expected_lock_path):
        raise EvidenceError("resource gate lock path differs from the selected physical GPU")
    for group in ("pre_lock_samples", "post_lock_samples"):
        samples = gate.get(group)
        if not isinstance(samples, list) or len(samples) != 2:
            raise EvidenceError(f"resource gate {group} must contain two samples")
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("physical_gpu") != selected:
                raise EvidenceError(f"resource gate {group} physical GPU differs")
            _validated_gpu_sample_values(sample, f"resource gate {group}")
            if not _gpu_is_launch_idle(sample, policy):
                raise EvidenceError(f"resource gate {group} exceeds the launch threshold")
        first_monotonic = _finite_record_number(
            samples[0], "monotonic_seconds", f"resource gate {group}[0]"
        )
        second_monotonic = _finite_record_number(
            samples[1], "monotonic_seconds", f"resource gate {group}[1]"
        )
        if second_monotonic - first_monotonic < float(
            policy["sample_interval_seconds"]
        ):
            raise EvidenceError(
                f"resource gate {group} does not prove the configured sample interval"
            )
    for group in ("host_pre_lock", "host_post_lock"):
        sample = gate.get(group)
        if not isinstance(sample, dict):
            raise EvidenceError(f"resource gate {group} is invalid")
        if _finite_record_number(sample, "available_host_memory_gib", group) < float(
            policy["minimum_available_host_memory_gib"]
        ):
            raise EvidenceError(f"resource gate {group} host memory is below threshold")
        if _finite_record_number(sample, "output_disk_free_gib", group) < float(
            policy["minimum_output_disk_free_gib"]
        ):
            raise EvidenceError(f"resource gate {group} disk space is below threshold")

    arguments = run_config.get("arguments")
    if not isinstance(arguments, dict):
        raise EvidenceError("run_config has no arguments for resource gate binding")
    try:
        argument_gate_path = Path(
            str(arguments["resource_gate_manifest"])
        ).expanduser().resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise EvidenceError("run_config resource gate argument is invalid") from exc
    if argument_gate_path != gate_path:
        raise EvidenceError("run_config resource gate path differs from the suite path")
    consumed = run_config.get("resource_gate")
    if not isinstance(consumed, dict):
        raise EvidenceError("run_config has no consumed resource gate evidence")
    expected_consumed = {
        "schema_version": RESOURCE_GATE_SCHEMA_VERSION,
        "path": str(gate_path),
        "sha256": gate_sha,
        "selected_physical_gpu": selected,
        "consumed_before_model_load": True,
    }
    for field, expected in expected_consumed.items():
        if consumed.get(field) != expected:
            raise EvidenceError(f"run_config consumed resource gate {field} differs")
    runtime_config = run_config.get("runtime")
    if not isinstance(runtime_config, dict) or runtime_config.get(
        "cuda_visible_devices"
    ) != str(selected):
        raise EvidenceError("run_config runtime does not prove the selected physical GPU")

    try:
        runtime_path = job.resource_runtime_path.resolve(strict=True)
        runtime_raw = runtime_path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read the resource runtime manifest: {exc}") from exc
    runtime_sha = hashlib.sha256(runtime_raw).hexdigest()
    runtime = _decode_json_object(runtime_raw, runtime_path)
    required_runtime = {
        "schema_version": RESOURCE_RUNTIME_SCHEMA_VERSION,
        "suite_id": job.suite_id,
        "job_id": job.job_id,
        "suite_config_sha256": job.suite_config_sha256,
        "job_config_sha256": job.job_config_sha256,
        "execution_fingerprint_sha256": job.execution_fingerprint_sha256,
        "selected_physical_gpu": selected,
        "resource_gate_sha256": gate_sha,
        "timed_out": False,
        "runner_exit_code": 0,
        "monitor_errors": [],
        "limits_passed": True,
    }
    for field, expected in required_runtime.items():
        if runtime.get(field) != expected:
            raise EvidenceError(f"resource runtime {field} differs from completed evidence")
    timeout_field = (
        "sentinel_timeout_hours"
        if int(job.tensor_scope["expected_selected_tensors"]) == 1
        else "three_tensor_pair_timeout_hours"
    )
    expected_timeout_seconds = float(policy[timeout_field]) * 3600.0
    recorded_timeout_seconds = _finite_record_number(
        runtime, "timeout_seconds", "resource_runtime"
    )
    if recorded_timeout_seconds != expected_timeout_seconds:
        raise EvidenceError("resource runtime timeout differs from the policy")
    expected_monitor_interval = min(
        5.0, max(1.0, float(policy["sample_interval_seconds"]) / 6.0)
    )
    recorded_monitor_interval = _finite_record_number(
        runtime, "sample_interval_seconds", "resource_runtime"
    )
    if recorded_monitor_interval != expected_monitor_interval:
        raise EvidenceError("resource runtime monitor interval differs from the policy")
    if _finite_record_number(
        runtime, "maximum_gpu_memory_mib", "resource_runtime"
    ) != float(policy["maximum_gpu_memory_mib"]):
        raise EvidenceError("resource runtime GPU memory limit differs from the policy")
    if _finite_record_number(runtime, "maximum_rss_gib", "resource_runtime") != float(
        policy["maximum_rss_gib"]
    ):
        raise EvidenceError("resource runtime RSS limit differs from the policy")
    if runtime.get("child_rss_measurement") != "RUSAGE_CHILDREN.ru_maxrss upper bound":
        raise EvidenceError("resource runtime child RSS measurement differs")
    sample_count = _positive_record_integer(runtime, "sample_count", "resource_runtime")
    peak_gpu = _finite_record_number(runtime, "peak_gpu_memory_mib", "resource_runtime")
    peak_gpu_utilization = _finite_record_number(
        runtime, "peak_gpu_utilization_percent", "resource_runtime"
    )
    peak_rss = _finite_record_number(runtime, "child_max_rss_gib", "resource_runtime")
    samples = runtime.get("gpu_samples")
    if not isinstance(samples, list) or len(samples) != sample_count:
        raise EvidenceError("resource runtime GPU sample ledger is inconsistent")
    if any(
        not isinstance(sample, dict) or sample.get("physical_gpu") != selected
        for sample in samples
    ):
        raise EvidenceError("resource runtime samples differ from the selected physical GPU")
    sample_memories: list[int] = []
    sample_utilizations: list[int] = []
    for sample in samples:
        memory, utilization = _validated_gpu_sample_values(
            sample, "resource runtime sample"
        )
        sample_memories.append(memory)
        sample_utilizations.append(utilization)
    recomputed_peak_gpu = max(sample_memories)
    recomputed_peak_utilization = max(sample_utilizations)
    if peak_gpu != float(recomputed_peak_gpu):
        raise EvidenceError("resource runtime peak GPU memory differs from its samples")
    if peak_gpu_utilization != float(recomputed_peak_utilization):
        raise EvidenceError("resource runtime peak GPU utilization differs from its samples")
    if recomputed_peak_gpu > int(policy["maximum_gpu_memory_mib"]):
        raise EvidenceError("resource runtime peak GPU memory exceeds the policy")
    if peak_rss < 0.0 or peak_rss > float(policy["maximum_rss_gib"]):
        raise EvidenceError("resource runtime child RSS exceeds the policy")
    return {
        "selected_physical_gpu": selected,
        "cuda_visible_devices": str(selected),
        "gate_manifest": {
            "path": str(gate_path),
            "size_bytes": len(gate_raw),
            "sha256": gate_sha,
        },
        "runtime_manifest": {
            "path": str(runtime_path),
            "size_bytes": len(runtime_raw),
            "sha256": runtime_sha,
        },
        "sample_count": sample_count,
        "peak_gpu_memory_mib": peak_gpu,
        "peak_gpu_utilization_percent": peak_gpu_utilization,
        "child_max_rss_gib": peak_rss,
        "limits_passed": True,
    }


def _validate_covariance_audit(job: SuiteJob, selected_layers: Sequence[str]) -> None:
    path = job.output_dir / "covariance_psd_audit.csv"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise EvidenceError(f"cannot read {path}: {exc}") from exc
    if [row.get("layer") for row in rows] != list(selected_layers):
        raise EvidenceError("covariance_psd_audit.csv layer order/set differs from selected_layers")
    finite_fields = (
        "collected_diagonal_mean",
        "collected_min_eigenvalue",
        "collected_spectral_scale",
        "configured_damping",
        "configured_damping_ratio",
        "diagonal_shift",
        "diagonal_shift_relative",
        "final_min_eigenvalue",
        "final_spectral_scale",
        "float32_storage_floor_rtol",
        "original_min_eigenvalue",
        "original_min_relative",
        "original_spectral_scale",
        "psd_rejection_rtol",
    )
    for row in rows:
        layer = str(row.get("layer", ""))
        for field in finite_fields:
            _finite_record_number(row, field, f"covariance_psd_audit.csv {layer}")
        if _positive_record_integer(
            row, "spectrum_decomposition_count", f"covariance_psd_audit.csv {layer}"
        ) != 1:
            raise EvidenceError(f"covariance spectrum was decomposed more than once for {layer}")
        if row.get("downstream_covariance_binding") != "immutable_prevalidated_input_covariance":
            raise EvidenceError(f"covariance binding differs for {layer}")


def _validate_global_allocator(
    run_config: Mapping[str, Any],
    arguments: Mapping[str, Any],
    endpoint_rows: Mapping[str, Mapping[str, Any]] | None = None,
    audited_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    if arguments.get("rate_allocation") != "global_exact":
        return
    allocator = run_config.get("rate_allocator")
    if not isinstance(allocator, dict):
        raise EvidenceError("global_exact job has no rate_allocator evidence")
    expected = {
        "mode": "global_exact",
        "selection_source": "global_exact_canonical_layout_pareto_frontier",
        "strict_file_byte_feasible": True,
        "frontier_coarsening_events": 0,
    }
    for field, value in expected.items():
        if allocator.get(field) != value:
            raise EvidenceError(
                f"global_exact allocator {field} differs: {allocator.get(field)!r} != {value!r}"
            )
    cap = _positive_record_integer(allocator, "q_l_cap_natural_file_bytes", "rate_allocator")
    selected = _positive_record_integer(
        allocator, "selected_qsl_natural_file_bytes", "rate_allocator"
    )
    unused = _finite_record_number(
        allocator, "unused_natural_bytes_before_tail_padding", "rate_allocator"
    )
    if selected > cap or unused != float(cap - selected):
        raise EvidenceError("global_exact allocator byte ledger does not close")
    if _positive_record_integer(
        allocator, "full_serializer_cross_checks", "rate_allocator"
    ) < 2:
        raise EvidenceError("global_exact allocator lacks both serializer cross-checks")
    qsl_cost = _finite_record_number(allocator, "selected_hessian_cost", "rate_allocator")
    two_stage = arguments.get("two_stage_selection") is True
    if two_stage:
        selection = allocator.get("two_stage_selection")
        if not isinstance(selection, dict):
            raise EvidenceError("two-stage global allocator has no selection evidence")
        expected_selection = {
            "enabled": True,
            "rerank_metric": "validation_nll",
            "test_split_reserved_until_after_selection": True,
        }
        for field, value in expected_selection.items():
            if selection.get(field) != value:
                raise EvidenceError(
                    f"two-stage global allocator {field} differs"
                )
        if selection.get("selection_split") != arguments.get("selection_split"):
            raise EvidenceError(
                "two-stage global allocator selection split differs from the job"
            )
        if _positive_record_integer(
            selection, "proxy_top_k", "rate_allocator two_stage_selection"
        ) != int(arguments.get("selection_top_k", 0)):
            raise EvidenceError(
                "two-stage global allocator proxy top-K differs from the job"
            )
        qsl_proxy_cost = _finite_record_number(
            allocator, "proxy_best_hessian_cost", "rate_allocator"
        )
    else:
        qsl_proxy_cost = qsl_cost

    if (endpoint_rows is None) != (audited_artifacts is None):
        raise EvidenceError("allocator external cross-check inputs are incomplete")
    if endpoint_rows is not None and audited_artifacts is not None:
        try:
            ql_natural = int(audited_artifacts["Q+L"]["natural_file_bytes"])
            qsl_natural = int(audited_artifacts["Q+S+L_QL_budget"]["natural_file_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceError(
                "global allocator artifacts lack Q+L/QSL natural-byte evidence"
            ) from exc
        selected_alias = _positive_record_integer(
            allocator, "selected_natural_file_bytes", "rate_allocator"
        )
        if cap != ql_natural:
            raise EvidenceError("global allocator cap differs from the Q+L artifact natural bytes")
        if selected != selected_alias or selected != qsl_natural:
            raise EvidenceError(
                "global allocator selected natural bytes differ from the QSL artifact"
            )
        endpoint_qsl_cost = _finite_record_number(
            endpoint_rows["Q+S+L_QL_budget"],
            "hessian_cost",
            "strategy_endpoints.csv Q+S+L_QL_budget",
        )
        cost_tolerance = 1e-10 * max(1.0, abs(qsl_cost), abs(endpoint_qsl_cost))
        if not math.isclose(
            qsl_cost, endpoint_qsl_cost, rel_tol=1e-10, abs_tol=cost_tolerance
        ):
            raise EvidenceError(
                "global allocator selected Hessian cost differs from the QSL endpoint"
            )

    if arguments.get("include_global_single_component_controls") is not True:
        return
    reports = allocator.get("global_control_reports")
    if reports is None:
        # Backward-compatible reader for the first implementation draft.  New
        # runs emit the accurately named field below as well.
        reports = allocator.get("single_component_controls")
    if not isinstance(reports, dict):
        raise EvidenceError("global control run has no nested allocator reports")
    expected_controls = {
        "Q+S_OBS_global",
        "Q+L_global",
        "Q+S_OBS_or_L_global",
    }
    if set(reports) != expected_controls:
        raise EvidenceError(
            "global control allocator report set differs: "
            f"{sorted(reports)} != {sorted(expected_controls)}"
        )
    natural_match_available = (
        allocator.get("joint_control_natural_match_available") is True
    )
    recorded_validation_scope = allocator.get(
        "joint_validation_counterfactual_scope"
    )
    if two_stage and recorded_validation_scope is not None:
        expected_validation_scope = (
            "exact_natural_matched"
            if natural_match_available
            else "cap_best_under_shared_cap_descriptive"
        )
        if recorded_validation_scope != expected_validation_scope:
            raise EvidenceError(
                "joint validation counterfactual scope is inconsistent"
            )
    control_proxy_costs: dict[str, float] = {}
    for strategy in sorted(expected_controls):
        report = reports[strategy]
        context = f"rate_allocator global control {strategy}"
        if not isinstance(report, dict):
            raise EvidenceError(f"{context} report is not an object")
        expected_report = {
            "mode": "global_exact",
            "endpoint_label": strategy,
            "selection_source": (
                "global_exact_canonical_layout_exact_natural_dynamic_program"
                if strategy == "Q+S_OBS_or_L_global"
                and natural_match_available
                else "global_exact_canonical_layout_pareto_frontier"
            ),
            "strict_file_byte_feasible": True,
            "frontier_coarsening_events": 0,
            "fallback_policy": "forbidden_fail_closed",
        }
        for field, value in expected_report.items():
            if report.get(field) != value:
                raise EvidenceError(
                    f"{context} {field} differs: {report.get(field)!r} != {value!r}"
                )
        control_cap = _positive_record_integer(report, "q_l_cap_natural_file_bytes", context)
        control_selected = _positive_record_integer(
            report, "selected_natural_file_bytes", context
        )
        control_unused = _finite_record_number(
            report, "unused_natural_bytes_before_tail_padding", context
        )
        if control_cap != cap or control_selected > cap:
            raise EvidenceError(f"{context} does not use the shared Q+L byte cap")
        if control_unused != float(cap - control_selected):
            raise EvidenceError(f"{context} byte ledger does not close")
        if _positive_record_integer(report, "full_serializer_cross_checks", context) < 2:
            raise EvidenceError(f"{context} lacks both serializer cross-checks")
        control_cost = _finite_record_number(report, "selected_hessian_cost", context)
        control_proxy_costs[strategy] = (
            _finite_record_number(report, "proxy_best_hessian_cost", context)
            if two_stage
            else control_cost
        )
        if endpoint_rows is not None and audited_artifacts is not None:
            try:
                artifact_natural = int(audited_artifacts[strategy]["natural_file_bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EvidenceError(
                    f"{context} lacks external natural-byte artifact evidence"
                ) from exc
            if control_selected != artifact_natural:
                raise EvidenceError(
                    f"{context} selected natural bytes differ from the serialized artifact"
                )
            endpoint_cost = _finite_record_number(
                endpoint_rows[strategy],
                "hessian_cost",
                f"strategy_endpoints.csv {strategy}",
            )
            control_tolerance = 1e-10 * max(
                1.0, abs(control_cost), abs(endpoint_cost)
            )
            if not math.isclose(
                control_cost,
                endpoint_cost,
                rel_tol=1e-10,
                abs_tol=control_tolerance,
            ):
                raise EvidenceError(
                    f"{context} Hessian cost differs from strategy_endpoints.csv"
                )

    nojoint = reports["Q+S_OBS_or_L_global"]
    nojoint_cost = control_proxy_costs["Q+S_OBS_or_L_global"]
    if two_stage and isinstance(allocator.get("nojoint_cap_best_audit"), dict):
        cap_best_nojoint = allocator["nojoint_cap_best_audit"]
        if cap_best_nojoint.get(
            "selection_source"
        ) != "global_exact_canonical_layout_pareto_frontier":
            raise EvidenceError(
                "no-joint cap-best audit did not use the exact Pareto allocator"
            )
        nojoint_cost = _finite_record_number(
            cap_best_nojoint,
            "selected_hessian_cost",
            "rate_allocator nojoint_cap_best_audit",
        )
    pure_cost = min(
        control_proxy_costs[strategy]
        for strategy in ("Q+S_OBS_global", "Q+L_global")
    )
    if allocator.get("nonjoint_union_weakly_dominates_pure_controls") is not True:
        raise EvidenceError("global allocator does not prove no-joint weak dominance")
    recorded_pure_cost = _finite_record_number(
        allocator, "best_pure_control_hessian_cost", "rate_allocator"
    )
    heterogeneous_gain = _finite_record_number(
        allocator, "nonjoint_heterogeneous_gain_over_best_pure", "rate_allocator"
    )
    if allocator.get("joint_control_strategy") != "Q+S_OBS_or_L_global":
        raise EvidenceError("global allocator does not identify the no-joint counterfactual")
    if allocator.get("joint_control_weakly_dominated") is not True:
        raise EvidenceError("global allocator does not prove QSL weak dominance")
    incremental_gain = _finite_record_number(
        allocator, "joint_candidate_incremental_hessian_gain", "rate_allocator"
    )
    tolerance = 1e-10 * max(1.0, abs(qsl_proxy_cost), abs(nojoint_cost))
    if nojoint_cost > pure_cost + tolerance or not math.isclose(
        recorded_pure_cost, pure_cost, rel_tol=1e-10, abs_tol=tolerance
    ):
        raise EvidenceError("no-joint union does not weakly dominate the best pure control")
    if not math.isclose(
        heterogeneous_gain,
        pure_cost - nojoint_cost,
        rel_tol=1e-10,
        abs_tol=tolerance,
    ):
        raise EvidenceError("no-joint heterogeneous gain is inconsistent")
    selected_layers = run_config.get("selected_layers")
    if isinstance(selected_layers, list) and len(selected_layers) == 1 and not math.isclose(
        nojoint_cost, pure_cost, rel_tol=1e-10, abs_tol=tolerance
    ):
        raise EvidenceError("one-layer no-joint union differs from the best pure control")
    if qsl_proxy_cost > nojoint_cost + tolerance:
        raise EvidenceError("QSL cost is worse than its nested no-joint control")
    if not math.isclose(
        incremental_gain,
        nojoint_cost - qsl_proxy_cost,
        rel_tol=1e-10,
        abs_tol=tolerance,
    ):
        raise EvidenceError("joint-candidate incremental Hessian gain is inconsistent")

    if natural_match_available:
        required_natural = _positive_record_integer(
            allocator,
            "joint_control_required_natural_file_bytes",
            "rate_allocator",
        )
        if required_natural != selected:
            raise EvidenceError(
                "joint counterfactual natural-byte target differs from selected QSL"
            )
        if endpoint_rows is not None and audited_artifacts is not None:
            nojoint_natural = int(
                audited_artifacts["Q+S_OBS_or_L_global"]["natural_file_bytes"]
            )
            if nojoint_natural != selected:
                raise EvidenceError(
                    "joint counterfactual is not exactly natural-byte matched to QSL"
                )

    claim = allocator.get("joint_value_claim")
    if claim is not None:
        if not isinstance(claim, dict):
            raise EvidenceError("joint_value_claim is not an object")
        if claim.get("supported") is True and not natural_match_available:
            raise EvidenceError(
                "joint_value_claim is supported without an exact natural-byte match"
            )
        if endpoint_rows is not None and audited_artifacts is not None:
            qsl_row = endpoint_rows["Q+S+L_QL_budget"]
            nojoint_row = endpoint_rows["Q+S_OBS_or_L_global"]
            qsl_natural = int(
                audited_artifacts["Q+S+L_QL_budget"]["natural_file_bytes"]
            )
            nojoint_natural = int(
                audited_artifacts["Q+S_OBS_or_L_global"]["natural_file_bytes"]
            )
            same_natural = qsl_natural == nojoint_natural
            qsl_nll = _finite_record_number(
                qsl_row,
                "heldout_nll",
                "strategy_endpoints.csv Q+S+L_QL_budget",
            )
            nojoint_nll = _finite_record_number(
                nojoint_row,
                "heldout_nll",
                "strategy_endpoints.csv Q+S_OBS_or_L_global",
            )
            gain = nojoint_nll - qsl_nll
            expected_supported = (
                natural_match_available and same_natural and gain > 0.0
            )
            recorded_exact_search = claim.get(
                "exact_natural_match_search_succeeded"
            )
            if (
                recorded_exact_search is not None
                and recorded_exact_search is not natural_match_available
            ):
                raise EvidenceError(
                    "joint_value_claim exact-search status is inconsistent"
                )
            if claim.get("same_natural_file_bytes") is not same_natural:
                raise EvidenceError(
                    "joint_value_claim natural-byte equality is inconsistent"
                )
            recorded_gain = _finite_record_number(
                claim,
                "qsl_test_nll_gain_over_nojoint",
                "joint_value_claim",
            )
            gain_tolerance = 1e-12 * max(1.0, abs(gain), abs(recorded_gain))
            if not math.isclose(
                recorded_gain,
                gain,
                rel_tol=1e-12,
                abs_tol=gain_tolerance,
            ):
                raise EvidenceError("joint_value_claim test NLL gain is inconsistent")
            if claim.get("supported") is not expected_supported:
                raise EvidenceError(
                    "joint_value_claim support flag is inconsistent with bytes/test NLL"
                )


def _csv_boolean(record: Mapping[str, Any], field: str, context: str) -> bool:
    value = record.get(field)
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise EvidenceError(f"{context} has invalid boolean {field}")


def _validate_two_stage_selection_evidence(
    job: SuiteJob,
    run_config: Mapping[str, Any],
    data: Mapping[str, Any],
    endpoint_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if job.effective_arguments.get("two_stage_selection") is not True:
        return None
    expected_splits = {
        "calibration": str(job.effective_arguments["calibration_split"]),
        "selection": str(job.effective_arguments["selection_split"]),
        "test": str(job.effective_arguments["test_split"]),
    }
    if data.get("role_splits") != expected_splits:
        raise EvidenceError("two-stage data role splits differ from the job")
    if data.get("test_reserved_until_after_validation_selection") is not True:
        raise EvidenceError(
            "two-stage data does not prove that test was reserved until selection"
        )
    for field in (
        "calibration_selection_identical_text_overlap_count",
        "identical_text_overlap_count",
        "selection_test_identical_text_overlap_count",
    ):
        if data.get(field) != 0:
            raise EvidenceError(f"two-stage data has non-zero {field}")
    for field in ("calib_digest", "selection_digest", "eval_digest"):
        value = data.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise EvidenceError(f"two-stage data has invalid {field}")
    if len(
        {
            str(data["calib_digest"]),
            str(data["selection_digest"]),
            str(data["eval_digest"]),
        }
    ) != 3:
        raise EvidenceError("two-stage token/text role digests are not distinct")

    baseline = run_config.get("allocation_validation_baseline_metrics")
    if not isinstance(baseline, dict):
        raise EvidenceError("two-stage run has no validation baseline metrics")
    for field in ("nll", "perplexity"):
        _finite_record_number(baseline, field, "allocation validation baseline")
    expected_validation_tokens = int(job.effective_arguments["selection_limit"]) * (
        int(job.effective_arguments["sequence_length"]) - 1
    )
    if _positive_record_integer(
        baseline, "tokens", "allocation validation baseline"
    ) != expected_validation_tokens:
        raise EvidenceError("allocation validation baseline token count differs")

    rerank_path = job.output_dir / "allocation_validation_rerank.csv"
    try:
        with rerank_path.open("r", encoding="utf-8", newline="") as handle:
            rerank_rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise EvidenceError(f"cannot read {rerank_path}: {exc}") from exc
    if not rerank_rows:
        raise EvidenceError("allocation_validation_rerank.csv is empty")
    dense_rows = [
        row
        for row in rerank_rows
        if row.get("strategy") == "dense_validation"
    ]
    if len(dense_rows) != 1 or dense_rows[0].get("proxy_rank") != "0":
        raise EvidenceError("allocation validation has an invalid dense baseline row")
    expected_strategies = {"Q+S+L_QL_budget"}
    if job.effective_arguments.get("include_global_single_component_controls") is True:
        expected_strategies.update(
            {
                "Q+S_OBS_global",
                "Q+L_global",
                "Q+S_OBS_or_L_global",
            }
        )
    candidate_rows = [
        row
        for row in rerank_rows
        if row.get("strategy") != "dense_validation"
    ]
    observed_strategies = {str(row.get("strategy", "")) for row in candidate_rows}
    if observed_strategies != expected_strategies:
        raise EvidenceError(
            "allocation validation strategy set differs from the global allocator"
        )
    allocator = run_config.get("rate_allocator")
    if not isinstance(allocator, dict):
        raise EvidenceError("two-stage run has no rate_allocator")
    selection = allocator.get("two_stage_selection")
    if not isinstance(selection, dict):
        raise EvidenceError("two-stage allocator selection evidence is missing")
    reports = selection.get("selection_reports")
    if not isinstance(reports, dict) or set(reports) != expected_strategies:
        raise EvidenceError("two-stage selection report set differs")
    selected_proxy_ranks: dict[str, int] = {}
    for strategy in sorted(expected_strategies):
        rows = [row for row in candidate_rows if row.get("strategy") == strategy]
        ranks = [
            _positive_record_integer(
                row, "proxy_rank", f"allocation validation {strategy}"
            )
            for row in rows
        ]
        if ranks != list(range(1, len(rows) + 1)):
            raise EvidenceError(
                f"allocation validation proxy ranks are not contiguous for {strategy}"
            )
        if len(rows) > int(job.effective_arguments["selection_top_k"]):
            raise EvidenceError(
                f"allocation validation exceeds the declared top-K for {strategy}"
            )
        selected_rows = [
            row
            for row in rows
            if _csv_boolean(
                row,
                "selected_by_validation",
                f"allocation validation {strategy}",
            )
        ]
        if len(selected_rows) != 1:
            raise EvidenceError(
                f"allocation validation must select exactly one row for {strategy}"
            )
        selected_row = selected_rows[0]
        selected_rank = int(selected_row["proxy_rank"])
        selected_proxy_ranks[strategy] = selected_rank
        report = reports[strategy]
        if not isinstance(report, dict):
            raise EvidenceError(f"two-stage selection report is invalid for {strategy}")
        if report.get("validation_selected_proxy_rank") != selected_rank:
            raise EvidenceError(
                f"two-stage selected proxy rank differs for {strategy}"
            )
        if report.get("validation_selected_allocation_digest") != selected_row.get(
            "allocation_digest"
        ):
            raise EvidenceError(
                f"two-stage allocation digest differs for {strategy}"
            )
        for field in (
            "hessian_cost",
            "natural_file_bytes",
            "validation_nll",
            "validation_perplexity",
            "validation_tokens",
            "validation_nll_delta",
        ):
            _finite_record_number(
                selected_row, field, f"allocation validation {strategy}"
            )
        endpoint = endpoint_rows[strategy]
        if endpoint.get("validation_selected_proxy_rank") != str(selected_rank):
            raise EvidenceError(
                f"endpoint validation proxy rank differs for {strategy}"
            )

    validation_window_path = (
        job.output_dir / "allocation_validation_window_nll.csv"
    )
    try:
        with validation_window_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            validation_windows = list(csv.DictReader(handle))
    except OSError as exc:
        raise EvidenceError(f"cannot read {validation_window_path}: {exc}") from exc
    if not validation_windows or any(
        row.get("evidence_role") != "allocation_validation"
        for row in validation_windows
    ):
        raise EvidenceError(
            "allocation validation windows are missing or have the wrong evidence role"
        )
    if any(row.get("strategy") in job.expected_strategies for row in validation_windows):
        raise EvidenceError(
            "allocation validation windows use final endpoint strategy labels"
        )

    endpoint_window_path = job.output_dir / "endpoint_window_nll.csv"
    try:
        with endpoint_window_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            endpoint_windows = list(csv.DictReader(handle))
    except OSError as exc:
        raise EvidenceError(f"cannot read {endpoint_window_path}: {exc}") from exc
    if not endpoint_windows or any(
        row.get("evidence_role") != "final_test" for row in endpoint_windows
    ):
        raise EvidenceError("endpoint windows are not exclusively final-test evidence")
    if any("__proxy_rank_" in str(row.get("strategy", "")) for row in endpoint_windows):
        raise EvidenceError("validation proxy labels leaked into final-test evidence")
    observed_test_strategies = {
        str(row.get("strategy", "")) for row in endpoint_windows
    }
    expected_test_strategies = {"dense", *job.expected_strategies}
    if observed_test_strategies != expected_test_strategies:
        raise EvidenceError("final-test strategy set differs from the suite contract")

    layer_count = int(job.tensor_scope["expected_selected_tensors"])
    assignment_fields = (
        "q_bits_by_layer",
        "q_quantizers_by_layer",
        "q_group_sizes_by_layer",
        "lowrank_factor_bits_by_layer",
    )
    for strategy, row in endpoint_rows.items():
        for field in assignment_fields:
            try:
                assignment = json.loads(str(row.get(field, "")))
            except json.JSONDecodeError as exc:
                raise EvidenceError(
                    f"strategy_endpoints.csv {strategy} has invalid {field}"
                ) from exc
            if not isinstance(assignment, dict) or len(assignment) != layer_count:
                raise EvidenceError(
                    f"strategy_endpoints.csv {strategy} {field} does not cover every layer"
                )

    return {
        "role_splits": expected_splits,
        "validation_tokens": expected_validation_tokens,
        "selected_proxy_ranks": selected_proxy_ranks,
        "test_reserved_until_after_validation_selection": True,
        "final_test_strategy_count": len(observed_test_strategies),
    }


def _protocol_window_ids(role: str, *, seed: int | None, count: int) -> list[str]:
    label = f"seed-{seed}" if seed is not None else "fixed"
    return [f"{role}/{label}/{index:03d}" for index in range(count)]


def _validate_protocol_run_evidence(job: SuiteJob, data: Mapping[str, Any]) -> dict[str, Any]:
    protocol = data.get("protocol")
    if not isinstance(protocol, dict):
        raise EvidenceError("run_config data has no protocol consumption evidence")
    required = {
        "consumed",
        "manifest_sha256",
        "schema_version",
        "selected_seed",
        "evaluation_role",
        "calibration_window_ids",
        "evaluation_window_ids",
        "calibration_token_sha256",
        "evaluation_token_sha256",
        "calibration_window_count",
        "evaluation_window_count",
        "window_token_length",
    }
    missing = required.difference(protocol)
    if missing:
        raise EvidenceError(f"run_config protocol evidence lacks fields: {sorted(missing)}")
    if protocol.get("consumed") is not True:
        raise EvidenceError("run_config does not prove that the protocol manifest was consumed")
    expected_scalars = {
        "manifest_sha256": job.protocol_manifest_sha256,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "selected_seed": job.protocol_seed,
        "evaluation_role": job.protocol_eval_role,
        "calibration_window_count": int(job.effective_arguments["calib_limit"]),
        "evaluation_window_count": int(job.effective_arguments["eval_limit"]),
        "window_token_length": int(job.effective_arguments["sequence_length"]),
    }
    for name, expected in expected_scalars.items():
        if protocol.get(name) != expected:
            raise EvidenceError(
                f"run_config protocol {name} differs from the declared job: {protocol.get(name)!r} != {expected!r}"
            )
    calibration_ids = protocol.get("calibration_window_ids")
    evaluation_ids = protocol.get("evaluation_window_ids")
    if not isinstance(calibration_ids, list) or not all(isinstance(item, str) for item in calibration_ids):
        raise EvidenceError("run_config protocol calibration_window_ids is invalid")
    if not isinstance(evaluation_ids, list) or not all(isinstance(item, str) for item in evaluation_ids):
        raise EvidenceError("run_config protocol evaluation_window_ids is invalid")
    if calibration_ids != _protocol_window_ids(
        "calibration", seed=job.protocol_seed, count=int(job.effective_arguments["calib_limit"])
    ):
        raise EvidenceError("run_config protocol calibration window order/identity differs")
    if evaluation_ids != _protocol_window_ids(
        str(job.protocol_eval_role), seed=None, count=int(job.effective_arguments["eval_limit"])
    ):
        raise EvidenceError("run_config protocol evaluation window order/identity differs")
    for name in ("calibration_token_sha256", "evaluation_token_sha256"):
        value = protocol.get(name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise EvidenceError(f"run_config protocol {name} is not a lowercase SHA-256 digest")
    # Preserve the complete consumer provenance in the evidence hash.  The
    # fields above are mandatory; additional dataset/model/tokenizer identity
    # fields emitted by the protocol consumer must remain tamper-evident too.
    return dict(protocol)


def _validate_protocol_activation_sampling(
    job: SuiteJob,
    data: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    sampling = data.get("protocol_activation_sampling")
    if not isinstance(sampling, dict):
        raise EvidenceError("run_config data has no protocol activation-sampling evidence")
    required = {
        "policy",
        "calibration_window_ids",
        "calibration_window_count",
        "total_token_rows",
        "sampled_rows_per_selected_tensor",
        "all_calibration_windows_traversed",
    }
    missing = required.difference(sampling)
    if missing:
        raise EvidenceError(
            f"protocol activation-sampling evidence lacks fields: {sorted(missing)}"
        )
    if sampling.get("policy") != PROTOCOL_ACTIVATION_SAMPLING_POLICY:
        raise EvidenceError("protocol activation-sampling policy differs")
    if sampling.get("calibration_window_ids") != protocol.get("calibration_window_ids"):
        raise EvidenceError("protocol activation-sampling calibration window identities differ")
    calibration_count = int(job.effective_arguments["calib_limit"])
    sequence_length = int(job.effective_arguments["sequence_length"])
    total_rows = calibration_count * sequence_length
    sample_rows = min(
        int(job.effective_arguments["selector_activation_sample_rows"]), total_rows
    )
    expected = {
        "calibration_window_count": calibration_count,
        "total_token_rows": total_rows,
        "sampled_rows_per_selected_tensor": sample_rows,
        "all_calibration_windows_traversed": True,
    }
    for field, value in expected.items():
        if sampling.get(field) != value:
            raise EvidenceError(
                f"protocol activation-sampling {field} differs: "
                f"{sampling.get(field)!r} != {value!r}"
            )
    return dict(sampling)


def _validate_protocol_model_binding(
    job: SuiteJob, run_config: Mapping[str, Any]
) -> dict[str, Any]:
    binding = run_config.get("protocol_model_binding")
    if not isinstance(binding, dict):
        raise EvidenceError("run_config has no protocol_model_binding evidence")
    revision = str(job.effective_arguments.get("revision", ""))
    required_scalars = {
        "expected_model_id": job.model_declared,
        "expected_snapshot_commit": revision,
        "requested_revision": revision,
        "resolved_model_commit_hash": revision,
        "model_name_or_path": job.model_declared,
        "tokenizer_name_or_path": job.model_declared,
        "model_class": "GPTNeoXForCausalLM",
        "tokenizer_class": "GPTNeoXTokenizerFast",
        "validated": True,
    }
    for field, expected in required_scalars.items():
        if binding.get(field) != expected:
            raise EvidenceError(
                f"protocol_model_binding {field} differs: "
                f"{binding.get(field)!r} != {expected!r}"
            )
    config_sha = binding.get("model_config_sha256")
    if not isinstance(config_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", config_sha):
        raise EvidenceError("protocol_model_binding model_config_sha256 is invalid")

    tokenizer_commit = binding.get("resolved_tokenizer_commit_hash")
    attestation = binding.get("tokenizer_runtime_commit_attestation")
    if attestation == "exact":
        if tokenizer_commit != revision:
            raise EvidenceError(
                "protocol tokenizer exact runtime attestation has a different commit"
            )
    elif attestation == "runtime_field_unavailable_asset_sha_bound":
        if tokenizer_commit is not None:
            raise EvidenceError(
                "protocol tokenizer unavailable-field attestation must record a null runtime commit"
            )
    else:
        raise EvidenceError("protocol tokenizer runtime commit attestation is invalid")

    snapshot_files = binding.get("snapshot_files")
    if not isinstance(snapshot_files, list) or len(snapshot_files) != len(
        PROTOCOL_FROZEN_HF_FILE_SHA256
    ):
        raise EvidenceError("protocol_model_binding snapshot_files is incomplete")
    observed_names = [
        item.get("filename") if isinstance(item, dict) else None for item in snapshot_files
    ]
    if observed_names != list(PROTOCOL_FROZEN_HF_FILE_SHA256):
        raise EvidenceError("protocol_model_binding snapshot file order/set differs")
    for item in snapshot_files:
        assert isinstance(item, dict)
        filename = str(item["filename"])
        if item.get("sha256") != PROTOCOL_FROZEN_HF_FILE_SHA256[filename]:
            raise EvidenceError(f"protocol snapshot SHA-256 differs for {filename}")
        if item.get("snapshot_commit") != revision:
            raise EvidenceError(f"protocol snapshot commit differs for {filename}")
        size_bytes = item.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise EvidenceError(f"protocol snapshot size is invalid for {filename}")

    identity = run_config.get("model_identity")
    if not isinstance(identity, dict):
        raise EvidenceError("run_config has no model_identity evidence")
    for field in (
        "requested_revision",
        "resolved_model_commit_hash",
        "resolved_tokenizer_commit_hash",
        "model_config_sha256",
        "model_name_or_path",
        "tokenizer_name_or_path",
    ):
        if identity.get(field) != binding.get(field):
            raise EvidenceError(
                f"protocol_model_binding {field} differs from run_config model_identity"
            )
    return dict(binding)


def _validate_protocol_numerical_path(
    job: SuiteJob,
    run_config: Mapping[str, Any],
    data: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if data.get("content_disjoint") is not True:
        raise EvidenceError("protocol data must be content_disjoint=true")
    if data.get("calib_digest") != protocol.get("calibration_token_sha256"):
        raise EvidenceError("protocol data calib_digest differs from consumed tokens")
    if data.get("eval_digest") != protocol.get("evaluation_token_sha256"):
        raise EvidenceError("protocol data eval_digest differs from consumed tokens")

    consumer = run_config.get("protocol_consumer")
    expected_consumer = {
        "version": PROTOCOL_SCHEMA_VERSION,
        "direct_token_tensor_input": True,
        "text_join_or_retokenization": False,
        "token_repetition": False,
    }
    if consumer != expected_consumer:
        raise EvidenceError("run_config protocol_consumer differs from the direct-token contract")

    validation_ids = protocol.get("validation_window_ids")
    test_ids = protocol.get("test_window_ids")
    if not isinstance(validation_ids, list) or not all(
        isinstance(item, str) for item in validation_ids
    ):
        raise EvidenceError("protocol validation_window_ids is invalid")
    if not isinstance(test_ids, list) or not all(isinstance(item, str) for item in test_ids):
        raise EvidenceError("protocol test_window_ids is invalid")
    calibration_count = int(job.effective_arguments["calib_limit"])
    evaluation_count = int(job.effective_arguments["eval_limit"])
    if len(validation_ids) != calibration_count:
        raise EvidenceError("protocol validation window count differs from the frozen screen")
    if len(test_ids) != evaluation_count or test_ids != protocol.get("evaluation_window_ids"):
        raise EvidenceError("protocol fixed test windows differ from endpoint evaluation windows")
    unique_reconstructed = len(
        set(protocol["calibration_window_ids"]) | set(validation_ids) | set(test_ids)
    )
    expected_counts = {
        "covariance_calibration": calibration_count,
        "activation_risk_calibration": calibration_count,
        "endpoint_nll_evaluation": evaluation_count,
        "comfort_recovery_validation": (
            0 if bool(job.effective_arguments.get("skip_comfort")) else len(validation_ids)
        ),
        "reconstructed_available_unique": unique_reconstructed,
    }
    counts = data.get("protocol_numerical_path_window_counts")
    if counts != expected_counts:
        raise EvidenceError(
            "protocol numerical-path window counts differ from the declared execution path"
        )
    return {
        "protocol_consumer": dict(expected_consumer),
        "protocol_numerical_path_window_counts": dict(expected_counts),
    }


def _validate_protocol_endpoint_windows(job: SuiteJob, protocol: Mapping[str, Any]) -> dict[str, str]:
    path = job.output_dir / "endpoint_window_nll.csv"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise EvidenceError(f"cannot read {path}: {exc}") from exc
    expected_ids = list(protocol["evaluation_window_ids"])
    expected_strategies = ["dense", *job.expected_strategies]
    grouped: dict[str, list[dict[str, str]]] = {strategy: [] for strategy in expected_strategies}
    for row in rows:
        strategy = str(row.get("strategy", ""))
        if strategy not in grouped:
            raise EvidenceError(f"endpoint_window_nll.csv contains unexpected strategy {strategy!r}")
        grouped[strategy].append(row)
    if any(not strategy_rows for strategy_rows in grouped.values()):
        raise EvidenceError("endpoint_window_nll.csv does not cover every expected strategy")
    reference_hashes: list[str] | None = None
    for strategy in expected_strategies:
        strategy_rows = grouped[strategy]
        if [str(row.get("protocol_window_id", "")) for row in strategy_rows] != expected_ids:
            raise EvidenceError(f"endpoint_window_nll.csv protocol window order differs for {strategy}")
        hashes = [str(row.get("protocol_window_sha256", "")) for row in strategy_rows]
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
            raise EvidenceError(f"endpoint_window_nll.csv has an invalid protocol window digest for {strategy}")
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise EvidenceError(f"endpoint_window_nll.csv protocol window digests differ for {strategy}")
        for index, row in enumerate(strategy_rows):
            if row.get("protocol_role") != job.protocol_eval_role:
                raise EvidenceError(
                    f"endpoint_window_nll.csv protocol role differs for {strategy}"
                )
            if row.get("protocol_seed") not in (None, ""):
                raise EvidenceError(
                    f"endpoint_window_nll.csv fixed evaluation seed differs for {strategy}"
                )
            try:
                window_index = int(row.get("window_index", ""))
                token_count = int(row.get("tokens", ""))
            except (TypeError, ValueError) as exc:
                raise EvidenceError(f"endpoint_window_nll.csv has invalid indices/counts for {strategy}") from exc
            if window_index != index:
                raise EvidenceError(f"endpoint_window_nll.csv window_index differs for {strategy}")
            if token_count != int(protocol["window_token_length"]) - 1:
                raise EvidenceError(f"endpoint_window_nll.csv token count differs for {strategy}")
    assert reference_hashes is not None
    return dict(zip(expected_ids, reference_hashes))


def _validate_log_evidence(
    record: Mapping[str, Any],
    *,
    key: str,
    expected_path: Path,
    suite_root: Path,
) -> None:
    payload = record.get(key)
    if not isinstance(payload, dict):
        raise EvidenceError(f"suite job record has no {key} log evidence")
    relative = payload.get("path")
    if not isinstance(relative, str) or not relative:
        raise EvidenceError(f"suite job record has an invalid {key} log path")
    path = _safe_relative_file(suite_root, relative)
    if path.resolve() != expected_path.resolve():
        raise EvidenceError(f"suite job record {key} path differs from the declared job log")
    if not path.is_file():
        raise EvidenceError(f"missing {key} log: {path}")
    expected_size = payload.get("size_bytes")
    expected_sha = payload.get("sha256")
    raw = path.read_bytes()
    raw_matches = expected_size == len(raw) and expected_sha == hashlib.sha256(raw).hexdigest()
    normalized = raw.replace(b"\r\n", b"\n")
    portable_matches = (
        expected_size == len(normalized)
        and expected_sha == hashlib.sha256(normalized).hexdigest()
    )
    if not raw_matches and not portable_matches:
        raise EvidenceError(
            f"{key} log size/SHA-256 differs from the suite job record after CRLF normalization"
        )


def validate_runner_outputs(job: SuiteJob, *, require_suite_record: bool) -> dict[str, Any]:
    job_dir = job.output_dir
    if not job_dir.is_dir():
        raise EvidenceError(f"missing job output directory: {job_dir}")
    if (job_dir / "RUNNING").exists():
        raise EvidenceError(f"RUNNING marker remains in {job_dir}")
    if (job_dir / "FAILED").exists():
        raise EvidenceError(f"FAILED marker exists in {job_dir}")
    _require_nonempty_file(job_dir / "COMPLETED")
    for relative in job.expected_outputs:
        _require_nonempty_file(_safe_relative_file(job_dir, relative))
    expected_output_files: dict[str, dict[str, Any]] | None = None
    if job.protocol_manifest_consumed:
        expected_output_files = {}
        for relative in ("COMPLETED", *job.expected_outputs):
            path = _safe_relative_file(job_dir, relative)
            expected_output_files[relative] = {
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }

    run_config = _read_json(job_dir / "run_config.json")
    arguments = run_config.get("arguments")
    if not isinstance(arguments, dict):
        raise EvidenceError("run_config.json has no arguments object")
    for name, expected in job.effective_arguments.items():
        if name in TRANSFORMED_RUNTIME_ARGUMENTS or (name == "revision" and expected == ""):
            continue
        if name not in arguments or not _value_matches(arguments[name], expected):
            raise EvidenceError(
                f"run_config argument {name!r} does not match the declared job: {arguments.get(name)!r} != {expected!r}"
            )
    if str(run_config.get("model")) != job.model_argument:
        raise EvidenceError("run_config model does not match the resolved model argument")
    model_snapshot_evidence = _validate_model_snapshot_evidence(job, run_config)
    selected_layers = run_config.get("selected_layers")
    expected_count = int(job.tensor_scope["expected_selected_tensors"])
    if not isinstance(selected_layers, list) or len(selected_layers) != expected_count:
        raise EvidenceError(
            f"selected tensor count differs from declared scope: {len(selected_layers) if isinstance(selected_layers, list) else 'invalid'} != {expected_count}"
        )
    requested_layers = set(map(int, job.tensor_scope["layers"]))
    requested_modules = set(map(str, job.tensor_scope["module_types"]))
    for layer_name in selected_layers:
        if not isinstance(layer_name, str):
            raise EvidenceError("selected_layers contains a non-string name")
        if _layer_index(layer_name) not in requested_layers or layer_name.rsplit(".", 1)[-1] not in requested_modules:
            raise EvidenceError(f"selected tensor falls outside the declared scope: {layer_name}")

    actual_eval_tokens = run_config.get("actual_eval_tokens")
    if isinstance(actual_eval_tokens, bool) or not isinstance(actual_eval_tokens, int) or actual_eval_tokens <= 0:
        raise EvidenceError("run_config has no positive actual_eval_tokens")
    activations = run_config.get("activation_counts")
    if not isinstance(activations, dict) or set(activations) != set(selected_layers):
        raise EvidenceError("activation_counts do not exactly cover selected_layers")
    activation_values = list(activations.values())
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in activation_values):
        raise EvidenceError("activation_counts contains a non-positive count")
    expected_eval_tokens = int(job.effective_arguments["eval_limit"]) * (
        int(job.effective_arguments["sequence_length"]) - 1
    )
    if actual_eval_tokens != expected_eval_tokens:
        raise EvidenceError(
            f"eval token count differs: {actual_eval_tokens} != {expected_eval_tokens}"
        )
    expected_calibration_tokens = int(job.effective_arguments["calib_limit"]) * int(
        job.effective_arguments["sequence_length"]
    )
    if any(value != expected_calibration_tokens for value in activation_values):
        raise EvidenceError(
            "calibration activation counts do not equal the declared window count x length"
        )

    run_sources = run_config.get("source_snapshot")
    if not isinstance(run_sources, dict):
        raise EvidenceError("run_config has no numerical source snapshot")
    if job.protocol_manifest_consumed and set(run_sources) != set(job.numerical_source_snapshot):
        raise EvidenceError("run_config protocol numerical source set differs from the suite contract")
    for name, expected_source in job.numerical_source_snapshot.items():
        actual_source = run_sources.get(name)
        if not isinstance(actual_source, dict) or actual_source.get("sha256") != expected_source["sha256"]:
            raise EvidenceError(f"run_config numerical source hash differs for {name}")

    manifest = _read_json(job_dir / "artifact_manifest.json")
    if manifest.get("production_backend") is not False:
        raise EvidenceError("research artifacts must not be labelled as a production backend")
    scope = manifest.get("scope")
    if scope != run_config.get("payload_scope"):
        raise EvidenceError("artifact scope differs from run_config payload_scope")
    if manifest.get("serialized_rate_cap_enforced") is not True:
        raise EvidenceError("serialized rate cap was not enforced")
    reference = manifest.get("reference")
    if not isinstance(reference, dict):
        raise EvidenceError("artifact manifest has no reference record")
    reference_evidence = _validate_artifact_file(job_dir, reference, "path", "file_bytes")
    strategies = manifest.get("strategies")
    if not isinstance(strategies, list):
        raise EvidenceError("artifact manifest strategies must be a list")
    names = [entry.get("strategy") for entry in strategies if isinstance(entry, dict)]
    if names != list(job.expected_strategies):
        raise EvidenceError("artifact manifest strategy order/set differs from the suite contract")
    strategy_evidence: dict[str, dict[str, Any]] = {}
    for entry in strategies:
        if not isinstance(entry, dict):
            raise EvidenceError("artifact manifest has a non-object strategy entry")
        strategy = str(entry["strategy"])
        if not math.isclose(float(entry.get("target_ratio", "nan")), job.target_rate, rel_tol=0.0, abs_tol=1e-12):
            raise EvidenceError(f"artifact target rate differs for {strategy}")
        strategy_evidence[strategy] = _validate_artifact_file(
            job_dir, entry, "artifact_path", "artifact_file_bytes"
        )
    _validate_strategy_csv(job)
    audited_strategy_evidence = _validate_physical_rates(
        job,
        strategies,
        reference_evidence,
        strategy_evidence,
    )
    endpoint_rows = _read_strategy_table(job, "strategy_endpoints.csv")
    _validate_global_allocator(
        run_config,
        arguments,
        endpoint_rows=endpoint_rows,
        audited_artifacts=audited_strategy_evidence,
    )
    endpoint_identity_evidence = _validate_endpoint_identities(
        job, endpoint_rows, audited_strategy_evidence
    )
    # Protocol evidence historically persists the recomputed physical-rate
    # fields in its evidence hash.  Legacy scalability records persist only
    # path/size/SHA; validate the richer ledger without silently invalidating
    # those already-published record hashes.
    if job.protocol_manifest_consumed or model_snapshot_evidence is not None:
        strategy_evidence = audited_strategy_evidence
    _validate_finite_endpoint_metrics(job)
    _validate_covariance_audit(job, selected_layers)

    data = run_config.get("data")
    runtime = run_config.get("runtime")
    if not isinstance(data, dict) or not isinstance(runtime, dict):
        raise EvidenceError("run_config is missing data/runtime provenance")
    resource_evidence = _validate_resource_evidence(job, run_config)
    if data.get("content_disjoint") is not True:
        raise EvidenceError("suite data must be content_disjoint=true")
    if data.get("fallback_allowed") is not False:
        raise EvidenceError("suite data must prove fallback_allowed=false")
    if data.get("identical_text_overlap_count") != 0:
        raise EvidenceError("suite data contains calibration/evaluation text overlap")
    two_stage_evidence = _validate_two_stage_selection_evidence(
        job,
        run_config,
        data,
        endpoint_rows,
    )
    protocol_evidence: dict[str, Any] | None = None
    protocol_window_digests: dict[str, str] | None = None
    protocol_activation_sampling: dict[str, Any] | None = None
    protocol_model_binding: dict[str, Any] | None = None
    protocol_numerical_path: dict[str, Any] | None = None
    if job.protocol_manifest_consumed:
        protocol_model_binding = _validate_protocol_model_binding(job, run_config)
        protocol_evidence = _validate_protocol_run_evidence(job, data)
        protocol_numerical_path = _validate_protocol_numerical_path(
            job, run_config, data, protocol_evidence
        )
        protocol_activation_sampling = _validate_protocol_activation_sampling(
            job, data, protocol_evidence
        )
        protocol_window_digests = _validate_protocol_endpoint_windows(job, protocol_evidence)
    evidence = {
        "payload_scope": scope,
        "scope_claim": job.tensor_scope["claim_scope"],
        "selected_tensor_count": len(selected_layers),
        "selected_parameter_count": int(run_config.get("selected_parameter_count", 0)),
        "model_parameter_count": int(run_config.get("model_parameter_count", 0)),
        "actual_eval_tokens": actual_eval_tokens,
        "actual_calibration_activation_tokens_min": min(activation_values),
        "actual_calibration_activation_tokens_max": max(activation_values),
        "calib_text_count": int(data.get("calib_text_count", 0)),
        "eval_text_count": int(data.get("eval_text_count", 0)),
        "content_disjoint": data.get("content_disjoint"),
        "calib_digest": data.get("calib_digest"),
        "eval_digest": data.get("eval_digest"),
        "reference_artifact": reference_evidence,
        "strategy_artifacts": strategy_evidence,
        "runtime": {
            "python": runtime.get("python"),
            "platform": runtime.get("platform"),
            "torch": runtime.get("torch"),
            "transformers": runtime.get("transformers"),
            "datasets": runtime.get("datasets"),
            "numpy": runtime.get("numpy"),
            "cuda_available": runtime.get("cuda_available"),
            "cuda_device": runtime.get("cuda_device"),
        },
        "model_identity": run_config.get("model_identity"),
    }
    if model_snapshot_evidence is not None:
        evidence["model_snapshot"] = model_snapshot_evidence
    if resource_evidence is not None:
        evidence["resource"] = resource_evidence
    if endpoint_identity_evidence:
        evidence["endpoint_identities"] = endpoint_identity_evidence
    if two_stage_evidence is not None:
        evidence["two_stage_selection"] = two_stage_evidence
    if protocol_evidence is not None:
        evidence["protocol"] = protocol_evidence
        evidence["protocol_activation_sampling"] = protocol_activation_sampling
        evidence["protocol_model_binding"] = protocol_model_binding
        evidence.update(protocol_numerical_path or {})
        evidence["protocol_evaluation_window_sha256"] = protocol_window_digests
        evidence["expected_output_files"] = expected_output_files

    if require_suite_record:
        record = _read_json(job_dir / "_suite_job_record.json")
        required = {
            "schema_version": JOB_RECORD_SCHEMA_VERSION,
            "status": "COMPLETED",
            "exit_code": 0,
            "suite_config_sha256": job.suite_config_sha256,
            "job_config_sha256": job.job_config_sha256,
            "numerical_source_sha256": job.numerical_source_sha256,
            "execution_fingerprint_sha256": job.execution_fingerprint_sha256,
        }
        for key, expected in required.items():
            if record.get(key) != expected:
                raise EvidenceError(f"suite job record mismatch for {key}")
        if job.protocol_manifest_consumed and record.get("protocol_manifest_consumed") is not True:
            raise EvidenceError("suite job record does not prove protocol consumption")
        if job.protocol_manifest_consumed and record.get("seed_aggregation_allowed") is not True:
            raise EvidenceError("suite job record does not authorize seed aggregation after protocol proof")
        if record.get("evidence_sha256") != _object_sha256(evidence):
            raise EvidenceError("suite job record evidence hash does not match current files")
        suite_root = job.output_dir.parents[1]
        _validate_log_evidence(
            record,
            key="stdout",
            expected_path=job.stdout_path,
            suite_root=suite_root,
        )
        _validate_log_evidence(
            record,
            key="stderr",
            expected_path=job.stderr_path,
            suite_root=suite_root,
        )
    return evidence


def inspect_job(job: SuiteJob) -> JobInspection:
    state_exists = job.state_path.exists()
    output_exists = job.output_dir.exists()
    if not state_exists and not output_exists:
        return JobInspection("planned")
    if not state_exists:
        return JobInspection("invalid", "output exists without orchestrator state")
    try:
        state = _read_json(job.state_path)
    except EvidenceError as exc:
        return JobInspection("invalid", str(exc))
    status = state.get("status")
    if state.get("suite_config_sha256") != job.suite_config_sha256:
        return JobInspection("invalid", "state suite config hash differs", state=state)
    if state.get("job_config_sha256") != job.job_config_sha256:
        return JobInspection("invalid", "state job config hash differs", state=state)
    if state.get("execution_fingerprint_sha256") != job.execution_fingerprint_sha256:
        return JobInspection("invalid", "state numerical/config fingerprint differs", state=state)
    if status == "RUNNING":
        return JobInspection("running", "RUNNING state is fail-closed; quarantine it before retry", state=state)
    if status == "FAILED":
        return JobInspection("failed", "FAILED state is fail-closed; quarantine it before retry", state=state)
    if status != "COMPLETED" or state.get("exit_code") != 0:
        return JobInspection("invalid", "state is neither COMPLETED+exit0 nor an explicit terminal failure", state=state)
    try:
        evidence = validate_runner_outputs(job, require_suite_record=True)
    except EvidenceError as exc:
        return JobInspection("invalid", str(exc), state=state)
    if state.get("evidence_sha256") != _object_sha256(evidence):
        return JobInspection("invalid", "state evidence hash differs from current output files", state=state)
    try:
        record = _read_json(job.output_dir / "_suite_job_record.json")
    except EvidenceError as exc:
        return JobInspection("invalid", str(exc), state=state)
    for key in ("stdout", "stderr"):
        if state.get(key) != record.get(key):
            return JobInspection("invalid", f"state {key} evidence differs from job record", state=state)
    return JobInspection("completed_valid", state=state, evidence=evidence)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _runtime_snapshot() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": status,
    }


def _job_manifest_entry(
    job: SuiteJob,
    inspection: JobInspection,
    *,
    command: Sequence[str],
) -> dict[str, Any]:
    common = job.effective_arguments
    evidence = inspection.evidence or {}
    actual_protocol_consumed = bool((evidence.get("protocol") or {}).get("consumed"))
    entry = {
        "job_id": job.job_id,
        "stage_id": job.stage_id,
        "lane": job.lane,
        "evidence_role": job.evidence_role,
        "protocol_manifest_consumed": actual_protocol_consumed if job.protocol_manifest_consumed else False,
        "seed_aggregation_allowed": (
            actual_protocol_consumed if job.protocol_manifest_consumed else job.seed_aggregation_allowed
        ),
        "data_window_independence": job.data_window_independence,
        "model_declared": job.model_declared,
        "model_argument": job.model_argument,
        "model_scale": job.model_scale,
        "model_availability": job.model_availability,
        "availability_note": job.availability_note,
        "model_override_env": job.model_override_env,
        "revision": job.revision or None,
        "seed": job.seed,
        "target_rate": job.target_rate,
        "tensor_scope": job.tensor_scope,
        "planned_calibration_tokens_per_selected_tensor": int(common["calib_limit"])
        * int(common["sequence_length"])
        * int(common["batch_size"]),
        "planned_eval_nll_tokens": int(common["eval_limit"])
        * (int(common["sequence_length"]) - 1)
        * int(common["batch_size"]),
        "output_dir": str(job.output_dir),
        "command": list(command),
        "suite_config_sha256": job.suite_config_sha256,
        "job_config_sha256": job.job_config_sha256,
        "numerical_source_sha256": job.numerical_source_sha256,
        "execution_fingerprint_sha256": job.execution_fingerprint_sha256,
        "status": inspection.status,
        "reason": inspection.reason,
        "exit_code": (inspection.state or {}).get("exit_code"),
        "actual": evidence or None,
    }
    if job.protocol_manifest_consumed:
        entry.update(
            {
                "protocol_manifest_consumption_planned": True,
                "seed_aggregation_planned": job.seed_aggregation_allowed,
                "protocol_manifest": job.protocol_manifest,
                "protocol_manifest_sha256": job.protocol_manifest_sha256,
                "protocol_seed": job.protocol_seed,
                "protocol_eval_role": job.protocol_eval_role,
            }
        )
    return entry


SUMMARY_FIELDS = (
    "job_id",
    "stage_id",
    "lane",
    "evidence_role",
    "protocol_manifest_consumed",
    "seed_aggregation_allowed",
    "data_window_independence",
    "model_declared",
    "model_argument",
    "model_scale",
    "model_availability",
    "seed",
    "target_rate",
    "tensor_scope",
    "expected_selected_tensors",
    "planned_calibration_tokens_per_tensor",
    "planned_eval_nll_tokens",
    "status",
    "exit_code",
    "actual_selected_tensors",
    "actual_calibration_tokens_min",
    "actual_calibration_tokens_max",
    "actual_eval_tokens",
    "artifact_scope",
    "reference_artifact_bytes",
    "cuda_device",
    "reason",
)


def write_suite_outputs(
    suite: SuiteDefinition,
    jobs: Sequence[SuiteJob],
    inspections: Mapping[str, JobInspection],
    *,
    python_executable: str,
    runner: Path,
) -> None:
    suite.output_root.mkdir(parents=True, exist_ok=True)
    protocol_interface = bool(
        suite.raw["evidence_contract"]["protocol_manifest_interface_supported"]
    )
    entries = [
        _job_manifest_entry(
            job,
            inspections[job.job_id],
            command=build_runner_command(job, python_executable=python_executable, runner=runner),
        )
        for job in jobs
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "suite_id": suite.suite_id,
        "updated_at": _utc_now(),
        "suite_config": str(suite.config_path),
        "suite_config_sha256": suite.config_sha256,
        "runner": str(runner),
        "numerical_source_snapshot": collect_numerical_source_snapshot(suite),
        "git": _git_snapshot(suite.repo_root),
        "orchestration_runtime": _runtime_snapshot(),
        "method_contract": {
            "one_process_per_model_seed_scope_rate": True,
            "expected_strategies": list(suite.expected_strategies),
            "artifact_scope_is_not_whole_model_unless_run_config_says_so": True,
            "resume_policy": "COMPLETED marker + exit_code 0 + matching suite/job/source hashes + all expected outputs and artifact hashes",
            "running_or_failed_policy": "fail_closed_manual_quarantine_required",
            "current_seed_policy": (
                "per-seed calibration source rows are disjoint; fixed test windows are shared for paired seed-level evaluation"
                if protocol_interface
                else "seed values share sequential data windows; repeated jobs are scalability checks and must not be pooled as independent evidence"
            ),
            "confirmatory_requires_protocol_manifest_consumed": True,
            "protocol_manifest_interface_supported_by_current_runner": protocol_interface,
        },
        "status_counts": {
            status: sum(entry["status"] == status for entry in entries)
            for status in ("planned", "running", "failed", "invalid", "completed_valid")
        },
        "jobs": entries,
    }
    if suite.raw.get("analysis_plan") is not None:
        manifest["analysis_plan"] = suite.raw["analysis_plan"]
    _write_json_atomic(suite.output_root / "suite_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    for entry in entries:
        actual = entry.get("actual") or {}
        reference = actual.get("reference_artifact") or {}
        runtime = actual.get("runtime") or {}
        rows.append(
            {
                "job_id": entry["job_id"],
                "stage_id": entry["stage_id"],
                "lane": entry["lane"],
                "evidence_role": entry["evidence_role"],
                "protocol_manifest_consumed": entry["protocol_manifest_consumed"],
                "seed_aggregation_allowed": entry["seed_aggregation_allowed"],
                "data_window_independence": entry["data_window_independence"],
                "model_declared": entry["model_declared"],
                "model_argument": entry["model_argument"],
                "model_scale": entry["model_scale"],
                "model_availability": entry["model_availability"],
                "seed": entry["seed"],
                "target_rate": entry["target_rate"],
                "tensor_scope": entry["tensor_scope"]["id"],
                "expected_selected_tensors": entry["tensor_scope"]["expected_selected_tensors"],
                "planned_calibration_tokens_per_tensor": entry["planned_calibration_tokens_per_selected_tensor"],
                "planned_eval_nll_tokens": entry["planned_eval_nll_tokens"],
                "status": entry["status"],
                "exit_code": entry["exit_code"],
                "actual_selected_tensors": actual.get("selected_tensor_count"),
                "actual_calibration_tokens_min": actual.get("actual_calibration_activation_tokens_min"),
                "actual_calibration_tokens_max": actual.get("actual_calibration_activation_tokens_max"),
                "actual_eval_tokens": actual.get("actual_eval_tokens"),
                "artifact_scope": actual.get("payload_scope"),
                "reference_artifact_bytes": reference.get("file_bytes"),
                "cuda_device": runtime.get("cuda_device"),
                "reason": entry["reason"],
            }
        )
    csv_path = suite.output_root / "suite_summary.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    counts = manifest["status_counts"]
    lines = [
        f"# {suite.suite_id}",
        "",
        "This is an orchestration/evidence summary, not a claim that planned jobs ran.",
        "",
        f"- Config SHA-256: `{suite.config_sha256}`",
        f"- Jobs: {len(jobs)}",
        f"- Valid completed: {counts['completed_valid']}",
        f"- Planned: {counts['planned']}",
        f"- Running (fail-closed): {counts['running']}",
        f"- Failed (fail-closed): {counts['failed']}",
        f"- Invalid (fail-closed): {counts['invalid']}",
        "",
        "## Stage matrix",
        "",
        "| Stage | Model | Availability | Evidence role | Seeds x rates | Tensor scope |",
        "|---|---|---|---|---:|---|",
    ]
    for stage in suite.stages:
        lines.append(
            "| {id} | {model} ({scale}) | {availability} | {role}; seed aggregation={aggregation} | {count} x {rates} | {scope} ({tensors} tensors) |".format(
                id=stage["id"],
                model=stage["model"],
                scale=stage["model_scale"],
                availability=stage["model_availability"],
                role=stage["evidence_role"],
                aggregation=str(stage["seed_aggregation_allowed"]).lower(),
                count=len(stage["seeds"]),
                rates=len(stage["rates"]),
                scope=stage["tensor_scope"]["id"],
                tensors=stage["tensor_scope"]["expected_selected_tensors"],
            )
        )
    protocol_note = (
        "Important: protocol-mode completion is accepted only when each job output proves the exact manifest SHA, seed, fixed ordered window identities and token digests. Statistical aggregation uses seed as the repeat unit; shared fixed test windows are paired diagnostics, not independent replicates."
        if protocol_interface
        else "Important: the current runner uses the same sequential data windows for every seed. Repeated seed jobs are scalability/reproducibility smoke checks only and must not be averaged or used as independent multi-seed evidence. A confirmatory label remains disabled until the numerical runner consumes the preregistered protocol manifest."
    )
    lines.extend(
        [
            "",
            "`suite_manifest.json` records commands, hashes, actual token counts, runtime/GPU provenance and physical artifact scope for completed jobs. Optional jobs are never executed unless selected explicitly or `--include-optional` is passed.",
            "",
            protocol_note,
            "",
        ]
    )
    (suite.output_root / "suite_summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _base_state(job: SuiteJob, command: Sequence[str]) -> dict[str, Any]:
    state = {
        "schema_version": JOB_RECORD_SCHEMA_VERSION,
        "job_id": job.job_id,
        "suite_id": job.suite_id,
        "evidence_role": job.evidence_role,
        "protocol_manifest_consumed": job.protocol_manifest_consumed,
        "seed_aggregation_allowed": job.seed_aggregation_allowed,
        "suite_config_sha256": job.suite_config_sha256,
        "job_config_sha256": job.job_config_sha256,
        "numerical_source_sha256": job.numerical_source_sha256,
        "execution_fingerprint_sha256": job.execution_fingerprint_sha256,
        "command": list(command),
        "orchestration_runtime": _runtime_snapshot(),
    }
    if job.protocol_manifest_consumed:
        state["protocol_manifest_consumption_planned"] = True
        state["protocol_manifest_consumed"] = False
        state["seed_aggregation_planned"] = job.seed_aggregation_allowed
        state["seed_aggregation_allowed"] = False
    return state


def _mark_failed(job: SuiteJob, payload: Mapping[str, Any]) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(job.output_dir / "FAILED", dict(payload))


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout: Any,
    stderr: Any,
    environment_overrides: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Narrow subprocess seam used by CPU-only lifecycle tests."""

    environment = os.environ.copy()
    source_root = (cwd / "src").resolve()
    if source_root.is_dir():
        inherited = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(source_root)
            if not inherited
            else os.pathsep.join((str(source_root), inherited))
        )
    if environment_overrides:
        environment.update({str(key): str(value) for key, value in environment_overrides.items()})
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )


def execute_job(
    suite: SuiteDefinition,
    job: SuiteJob,
    *,
    python_executable: str,
    runner: Path,
) -> dict[str, Any]:
    command = build_runner_command(job, python_executable=python_executable, runner=runner)
    job.state_path.parent.mkdir(parents=True, exist_ok=True)
    job.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    running = _base_state(job, command)
    running.update({"status": "RUNNING", "started_at": _utc_now(), "exit_code": None})
    _write_json_atomic(job.state_path, running)
    started = time.monotonic()
    exit_code = -1
    failure: str | None = None
    lease: ResourceLease | None = None
    try:
        lease = _acquire_resource_lease(suite, job)
        environment_overrides: dict[str, str] | None = None
        timeout_seconds: float | None = None
        monitor_stop: threading.Event | None = None
        monitor_samples: list[dict[str, Any]] = []
        monitor_errors: list[str] = []
        monitor_thread: threading.Thread | None = None
        runtime_started_at: str | None = None
        timed_out = False
        if lease is not None:
            environment_overrides = {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(lease.selected_gpu),
            }
            timeout_field = (
                "sentinel_timeout_hours"
                if int(job.tensor_scope["expected_selected_tensors"]) == 1
                else "three_tensor_pair_timeout_hours"
            )
            timeout_seconds = float((job.resource_policy or {})[timeout_field]) * 3600.0
            running["resource_gate"] = {
                "path": str(job.resource_gate_path),
                "sha256": _file_sha256(job.resource_gate_path),
                "selected_physical_gpu": lease.selected_gpu,
                "cuda_visible_devices": str(lease.selected_gpu),
            }
            _write_json_atomic(job.state_path, running)
            runtime_started_at = _utc_now()
            monitor_stop, monitor_samples, monitor_errors, monitor_thread = (
                _start_gpu_runtime_monitor(
                    lease.selected_gpu,
                    interval_seconds=min(
                        5.0,
                        max(
                            1.0,
                            float((job.resource_policy or {})["sample_interval_seconds"])
                            / 6.0,
                        ),
                    ),
                )
            )

        completed: subprocess.CompletedProcess[str] | None = None
        timeout_error: subprocess.TimeoutExpired | None = None
        try:
            with job.stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_handle, job.stderr_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as stderr_handle:
                completed = _run_process(
                    command,
                    cwd=suite.repo_root,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    environment_overrides=environment_overrides,
                    timeout_seconds=timeout_seconds,
                )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -9
            timeout_error = exc
        finally:
            if lease is not None:
                assert monitor_stop is not None and monitor_thread is not None
                monitor_stop.set()
                monitor_thread.join(timeout=30.0)
                if monitor_thread.is_alive():
                    monitor_errors.append("GPU resource monitor did not stop")
                child_rss_gib = _child_max_rss_gib()
                peak_gpu_memory = max(
                    (int(sample["memory_used_mib"]) for sample in monitor_samples),
                    default=-1,
                )
                peak_gpu_utilization = max(
                    (
                        int(sample["utilization_gpu_percent"])
                        for sample in monitor_samples
                    ),
                    default=-1,
                )
                policy = dict(job.resource_policy or {})
                if completed is not None:
                    exit_code = int(completed.returncode)
                limits_passed = (
                    not monitor_errors
                    and bool(monitor_samples)
                    and peak_gpu_memory <= int(policy["maximum_gpu_memory_mib"])
                    and child_rss_gib <= float(policy["maximum_rss_gib"])
                    and not timed_out
                    and exit_code == 0
                )
                runtime_evidence = {
                    "schema_version": RESOURCE_RUNTIME_SCHEMA_VERSION,
                    "suite_id": job.suite_id,
                    "job_id": job.job_id,
                    "suite_config_sha256": job.suite_config_sha256,
                    "job_config_sha256": job.job_config_sha256,
                    "execution_fingerprint_sha256": job.execution_fingerprint_sha256,
                    "resource_gate_sha256": _file_sha256(job.resource_gate_path),
                    "selected_physical_gpu": lease.selected_gpu,
                    "started_at": runtime_started_at,
                    "completed_at": _utc_now(),
                    "timeout_seconds": timeout_seconds,
                    "timed_out": timed_out,
                    "runner_exit_code": exit_code,
                    "sample_interval_seconds": min(
                        5.0,
                        max(1.0, float(policy["sample_interval_seconds"]) / 6.0),
                    ),
                    "sample_count": len(monitor_samples),
                    "gpu_samples": monitor_samples,
                    "monitor_errors": monitor_errors,
                    "peak_gpu_memory_mib": peak_gpu_memory,
                    "peak_gpu_utilization_percent": peak_gpu_utilization,
                    "child_max_rss_gib": child_rss_gib,
                    "child_rss_measurement": "RUSAGE_CHILDREN.ru_maxrss upper bound",
                    "maximum_gpu_memory_mib": int(policy["maximum_gpu_memory_mib"]),
                    "maximum_rss_gib": float(policy["maximum_rss_gib"]),
                    "limits_passed": limits_passed,
                }
                _write_json_atomic(job.resource_runtime_path, runtime_evidence)

        if timeout_error is not None:
            raise EvidenceError(
                f"numerical runner timed out after {timeout_seconds:.0f} seconds"
            ) from timeout_error
        assert completed is not None
        exit_code = int(completed.returncode)
        if exit_code != 0:
            raise EvidenceError(f"numerical runner exited with code {exit_code}")
        if lease is not None:
            runtime_evidence = _read_json(job.resource_runtime_path)
            if runtime_evidence.get("limits_passed") is not True:
                raise EvidenceError("numerical runner exceeded the resource policy")
        evidence = validate_runner_outputs(job, require_suite_record=False)
        elapsed = time.monotonic() - started
        evidence_hash = _object_sha256(evidence)
        record = _base_state(job, command)
        record.update(
            {
                "status": "COMPLETED",
                "exit_code": 0,
                "started_at": running["started_at"],
                "completed_at": _utc_now(),
                "elapsed_seconds": elapsed,
                "stdout": {
                    "path": job.stdout_path.relative_to(suite.output_root).as_posix(),
                    "size_bytes": job.stdout_path.stat().st_size,
                    "sha256": _file_sha256(job.stdout_path),
                },
                "stderr": {
                    "path": job.stderr_path.relative_to(suite.output_root).as_posix(),
                    "size_bytes": job.stderr_path.stat().st_size,
                    "sha256": _file_sha256(job.stderr_path),
                },
                "evidence_sha256": evidence_hash,
            }
        )
        if job.protocol_manifest_consumed:
            record["protocol_manifest_consumed"] = bool(
                (evidence.get("protocol") or {}).get("consumed")
            )
            record["seed_aggregation_allowed"] = bool(
                record["protocol_manifest_consumed"] and job.seed_aggregation_allowed
            )
        _write_json_atomic(job.output_dir / "_suite_job_record.json", record)
        validate_runner_outputs(job, require_suite_record=True)
        _write_json_atomic(job.state_path, record)
        return evidence
    except (OSError, EvidenceError, subprocess.SubprocessError) as exc:
        failure = str(exc)
        elapsed = time.monotonic() - started
        failed = _base_state(job, command)
        failed.update(
            {
                "status": "FAILED",
                "exit_code": exit_code,
                "started_at": running["started_at"],
                "failed_at": _utc_now(),
                "elapsed_seconds": elapsed,
                "failure": failure,
            }
        )
        _write_json_atomic(job.state_path, failed)
        _mark_failed(job, failed)
        raise EvidenceError(f"{job.job_id}: {failure}") from exc
    finally:
        if lease is not None:
            lease.release()


def select_jobs(
    jobs: Sequence[SuiteJob],
    *,
    job_id: str | None,
    stages: Sequence[str],
    include_optional: bool,
) -> tuple[list[SuiteJob], list[SuiteJob]]:
    known_ids = {job.job_id for job in jobs}
    known_stages = {job.stage_id for job in jobs}
    if job_id is not None and job_id not in known_ids:
        raise SuiteConfigError(f"unknown job id: {job_id}")
    unknown_stages = set(stages) - known_stages
    if unknown_stages:
        raise SuiteConfigError(f"unknown stage ids: {sorted(unknown_stages)}")
    chosen = [
        job
        for job in jobs
        if (job_id is None or job.job_id == job_id) and (not stages or job.stage_id in set(stages))
    ]
    runnable: list[SuiteJob] = []
    skipped_optional: list[SuiteJob] = []
    for job in chosen:
        if job.model_availability == "optional" and not include_optional and job_id is None:
            skipped_optional.append(job)
        else:
            runnable.append(job)
    return runnable, skipped_optional


def check_suite(
    suite: SuiteDefinition,
    jobs: Sequence[SuiteJob],
    selected: Sequence[SuiteJob],
) -> dict[str, Any]:
    inspections = {job.job_id: inspect_job(job) for job in jobs}
    invalid = [
        {"job_id": job.job_id, "status": inspections[job.job_id].status, "reason": inspections[job.job_id].reason}
        for job in selected
        if inspections[job.job_id].status in {"running", "failed", "invalid"}
    ]
    manifest_path = suite.output_root / "suite_manifest.json"
    persisted_errors: list[str] = []
    any_state = any(job.state_path.exists() or job.output_dir.exists() for job in jobs)
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                persisted_errors.append("persisted manifest schema differs")
            if manifest.get("suite_config_sha256") != suite.config_sha256:
                persisted_errors.append("persisted manifest config hash differs")
            entries = manifest.get("jobs")
            if not isinstance(entries, list):
                persisted_errors.append("persisted manifest jobs is invalid")
            else:
                by_id = {entry.get("job_id"): entry for entry in entries if isinstance(entry, dict)}
                if set(by_id) != {job.job_id for job in jobs}:
                    persisted_errors.append("persisted manifest job set differs")
                for job in selected:
                    entry = by_id.get(job.job_id, {})
                    if entry.get("job_config_sha256") != job.job_config_sha256:
                        persisted_errors.append(f"{job.job_id}: persisted job hash differs")
                    if entry.get("status") != inspections[job.job_id].status:
                        persisted_errors.append(f"{job.job_id}: persisted status is stale")
        except EvidenceError as exc:
            persisted_errors.append(str(exc))
        for relative in ("suite_summary.csv", "suite_summary.md"):
            path = suite.output_root / relative
            if not path.is_file() or path.stat().st_size <= 0:
                persisted_errors.append(f"missing persisted {relative}")
    elif any_state:
        persisted_errors.append("job state/output exists but suite_manifest.json is missing")
    return {
        "suite_id": suite.suite_id,
        "suite_config_sha256": suite.config_sha256,
        "selected_job_count": len(selected),
        "status_counts": {
            status: sum(inspections[job.job_id].status == status for job in selected)
            for status in ("planned", "running", "failed", "invalid", "completed_valid")
        },
        "invalid_jobs": invalid,
        "persisted_errors": persisted_errors,
        "ok": not invalid and not persisted_errors,
    }


def _dry_run_payload(
    selected: Sequence[SuiteJob],
    skipped_optional: Sequence[SuiteJob],
    *,
    python_executable: str,
    runner: Path,
    resume: bool,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for job in selected:
        inspection = inspect_job(job)
        if resume and inspection.status == "completed_valid":
            action = "skip_valid_completed"
        elif inspection.status == "planned":
            action = "would_execute"
        else:
            action = "fail_closed"
        jobs.append(
            {
                "job_id": job.job_id,
                "stage_id": job.stage_id,
                "availability": job.model_availability,
                "evidence_role": job.evidence_role,
                "protocol_manifest_consumed": job.protocol_manifest_consumed,
                "seed_aggregation_allowed": job.seed_aggregation_allowed,
                "seed": job.seed,
                "target_rate": job.target_rate,
                "tensor_scope": job.tensor_scope["id"],
                "status": inspection.status,
                "action": action,
                "command": build_runner_command(job, python_executable=python_executable, runner=runner),
            }
        )
    return {
        "dry_run": True,
        "writes_performed": False,
        "selected_jobs": jobs,
        "optional_jobs_not_selected": [job.job_id for job in skipped_optional],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "large_scale_hessian_suite_20260714.json",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--runner", type=Path, default=None)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--job", default=None, help="Execute/check one exact expanded job id")
    parser.add_argument("--stage", action="append", default=[], help="Select a stage; repeatable")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        suite = load_suite_definition(
            args.config,
            output_root_override=args.output_root,
            runner_override=args.runner,
        )
        recorded_model_arguments = load_recorded_model_arguments(suite) if args.check else None
        jobs = expand_jobs(suite, recorded_model_arguments=recorded_model_arguments)
        selected, skipped_optional = select_jobs(
            jobs,
            job_id=args.job,
            stages=args.stage,
            # Optional controls affect execution availability, not the audit of
            # already persisted evidence.  A check therefore covers them by
            # default; --job/--stage can still narrow the audited set.
            include_optional=args.include_optional or args.check,
        )
        if args.dry_run:
            print(
                json.dumps(
                    _dry_run_payload(
                        selected,
                        skipped_optional,
                        python_executable=args.python_executable,
                        runner=suite.runner,
                        resume=args.resume,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        if args.check:
            report = check_suite(suite, jobs, selected)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["ok"] else 1

        preflight = {job.job_id: inspect_job(job) for job in selected}
        blocked: list[str] = []
        to_run: list[SuiteJob] = []
        for job in selected:
            status = preflight[job.job_id].status
            if args.resume and status == "completed_valid":
                continue
            if status == "planned":
                to_run.append(job)
            else:
                blocked.append(f"{job.job_id}: {status}: {preflight[job.job_id].reason}")
        if blocked:
            raise EvidenceError(
                "preflight is fail-closed; move invalid/running/failed outputs and state aside before retry:\n"
                + "\n".join(blocked)
            )

        for job in to_run:
            try:
                execute_job(
                    suite,
                    job,
                    python_executable=args.python_executable,
                    runner=suite.runner,
                )
            except EvidenceError:
                inspections = {candidate.job_id: inspect_job(candidate) for candidate in jobs}
                write_suite_outputs(
                    suite,
                    jobs,
                    inspections,
                    python_executable=args.python_executable,
                    runner=suite.runner,
                )
                raise
            inspections = {candidate.job_id: inspect_job(candidate) for candidate in jobs}
            write_suite_outputs(
                suite,
                jobs,
                inspections,
                python_executable=args.python_executable,
                runner=suite.runner,
            )
        if not to_run:
            inspections = {candidate.job_id: inspect_job(candidate) for candidate in jobs}
            write_suite_outputs(
                suite,
                jobs,
                inspections,
                python_executable=args.python_executable,
                runner=suite.runner,
            )
        print(
            json.dumps(
                {
                    "suite_id": suite.suite_id,
                    "executed": [job.job_id for job in to_run],
                    "resumed_valid": [
                        job.job_id
                        for job in selected
                        if preflight[job.job_id].status == "completed_valid"
                    ],
                    "optional_jobs_not_selected": [job.job_id for job in skipped_optional],
                    "manifest": str(suite.output_root / "suite_manifest.json"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except (SuiteConfigError, EvidenceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
