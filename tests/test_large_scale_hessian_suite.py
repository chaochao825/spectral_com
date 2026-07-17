from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_large_scale_hessian_suite",
    REPO_ROOT / "scripts" / "run_large_scale_hessian_suite.py",
)
assert SPEC is not None and SPEC.loader is not None
SUITE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUITE
SPEC.loader.exec_module(SUITE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_config(tmp_path: Path, runner: Path) -> Path:
    config = {
        "schema_version": SUITE.SCHEMA_VERSION,
        "suite_id": "unit_suite",
        "description": "unit test",
        "evidence_contract": {
            "current_data_window_policy": "content-disjoint sequential windows are shared across seed values",
            "protocol_manifest_interface_supported": False,
            "multi_seed_aggregation_requires_consumed_protocol_manifest": True,
            "default_evidence_role": "scalability_smoke",
        },
        "output_root": str(tmp_path / "results"),
        "runner": str(runner),
        "expected_strategies": ["Q", "Q+L"],
        "expected_outputs": [
            "run_config.json",
            "strategy_endpoints.csv",
            "covariance_psd_audit.csv",
            "artifact_manifest.json",
            "artifact_payloads.csv",
        ],
        "common": {
            "calib_limit": 2,
            "eval_limit": 3,
            "sequence_length": 16,
            "batch_size": 1,
            "texts_per_batch_window": 2,
            "bits": 4,
            "emit_codec_artifacts": True,
            "enforce_serialized_rate_cap": True,
        },
        "stages": [
            {
                "id": "tiny",
                "lane": "A_post_training_no_backward",
                "evidence_role": "scalability_smoke",
                "protocol_manifest_consumed": False,
                "seed_aggregation_allowed": False,
                "data_window_independence": "shared_sequential_windows_not_independent_across_seeds",
                "model": "local/tiny",
                "model_scale": "tiny",
                "model_availability": "required",
                "availability_note": "unit-test fixture",
                "model_override_env": "UNIT_MODEL",
                "revision": "",
                "seeds": [17],
                "rates": [0.25],
                "tensor_scope": {
                    "id": "one_block_mlp",
                    "claim_scope": "two selected MLP tensors only",
                    "module_types": ["fc1", "fc2"],
                    "layers": [0],
                    "max_modules": 0,
                    "expected_selected_tensors": 2,
                },
            }
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _minimal_protocol_config(tmp_path: Path) -> Path:
    runner = tmp_path / "scripts" / "run_pretrained_hessian_repair_protocol_v2.py"
    required_sources = {
        runner,
        tmp_path / "scripts" / "confirmatory_protocol_windows.py",
        tmp_path / "scripts" / "run_pretrained_hessian_repair.py",
        tmp_path / "scripts" / "run_pretrained_llm_orthogonality.py",
        tmp_path / "src" / "llm_spectral_dynamics" / "structured" / "codec_artifact.py",
        tmp_path / "src" / "llm_spectral_dynamics" / "structured" / "hessian_repair.py",
        tmp_path / "src" / "llm_spectral_dynamics" / "structured" / "data.py",
    }
    for source in required_sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# fixture: {source.name}\n", encoding="utf-8")
    protocol = {
        "schema_version": SUITE.PROTOCOL_SCHEMA_VERSION,
        "status": "preregistered_data_split_manifest",
        "seeds": [17],
        "model": {
            "model_id": "local/tiny",
            "snapshot_commit": "unit-protocol-commit",
            "weights_loaded_by_this_script": False,
        },
        "tokenization": {
            "window_token_length": 16,
            "snapshot_commit": "unit-protocol-commit",
            "tokenizer_class": "GPTNeoXTokenizerFast",
        },
        "allocation_counts": {
            "calibration_windows_per_seed": 2,
            "validation_windows": 2,
            "test_windows": 3,
        },
        "windows": {
            "calibration_by_seed": {
                "17": [
                    {"window_id": f"calibration/seed-17/{index:03d}"}
                    for index in range(2)
                ]
            },
            "validation": [
                {"window_id": f"validation/fixed/{index:03d}"} for index in range(2)
            ],
            "test": [{"window_id": f"test/fixed/{index:03d}"} for index in range(3)],
        },
    }
    manifest_path = tmp_path / "protocol" / "protocol.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
    config_path = _minimal_config(tmp_path, runner)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["evidence_contract"] = {
        "current_data_window_policy": SUITE.PROTOCOL_WINDOW_POLICY,
        "protocol_manifest_interface_supported": True,
        "multi_seed_aggregation_requires_consumed_protocol_manifest": True,
        "default_evidence_role": "confirmatory",
    }
    payload["expected_outputs"].append("endpoint_window_nll.csv")
    payload["common"].update(
        {
            "skip_comfort": True,
            "skip_plots": True,
            "selector_activation_sample_rows": 5,
            "rate_tolerance": 0.01,
        }
    )
    payload["stages"][0].pop("model_override_env")
    payload["stages"][0].update(
        {
            "evidence_role": "confirmatory",
            "protocol_manifest_consumed": True,
            "seed_aggregation_allowed": True,
            "data_window_independence": SUITE.PROTOCOL_WINDOW_INDEPENDENCE,
            "protocol_manifest": "protocol/protocol.json",
            "protocol_manifest_sha256": _sha(manifest_path),
            "protocol_eval_role": "test",
            "revision": "unit-protocol-commit",
        }
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _make_fake_suite(tmp_path: Path) -> tuple[object, object]:
    runner = tmp_path / "fake_runner.py"
    runner.write_text("# numerical source used only for its digest\n", encoding="utf-8")
    config = _minimal_config(tmp_path, runner)
    definition = SUITE.load_suite_definition(config, repo_root=tmp_path)
    return definition, SUITE.expand_jobs(definition, environment={})[0]


def _write_fake_completed_output(job: object) -> None:
    output = job.output_dir
    output.mkdir(parents=True, exist_ok=True)
    selected = ["model.layers.0.mlp.fc1", "model.layers.0.mlp.fc2"]
    protocol_mode = bool(job.protocol_manifest_consumed)
    revision = str(job.effective_arguments.get("revision", ""))
    if protocol_mode:
        model_identity = {
            "requested_revision": revision,
            "resolved_model_commit_hash": revision,
            "resolved_tokenizer_commit_hash": None,
            "model_config_sha256": "3" * 64,
            "model_name_or_path": job.model_declared,
            "tokenizer_name_or_path": job.model_declared,
        }
    else:
        model_identity = {"resolved_model_commit_hash": "unit"}
    run_config = {
        "model": job.model_argument,
        "arguments": dict(job.effective_arguments),
        "selected_layers": selected,
        "selected_parameter_count": 64,
        "model_parameter_count": 128,
        "baseline_metrics": {"nll": 1.0, "perplexity": 2.718281828459045, "tokens": 45},
        "actual_eval_tokens": 45,
        "activation_counts": {name: 32 for name in selected},
        "source_snapshot": job.numerical_source_snapshot,
        "payload_scope": "selected_linear_weights_only",
        "data": {
            "calib_text_count": 4,
            "eval_text_count": 6,
            "content_disjoint": True,
            "fallback_allowed": False,
            "identical_text_overlap_count": 0,
        },
        "runtime": {
            "python": "3.11.0",
            "platform": "test",
            "torch": "test",
            "transformers": "test",
            "datasets": "test",
            "numpy": "test",
            "cuda_available": False,
            "cuda_device": None,
        },
        "model_identity": model_identity,
    }
    if protocol_mode:
        calibration_ids = SUITE._protocol_window_ids(
            "calibration", seed=job.seed, count=job.effective_arguments["calib_limit"]
        )
        evaluation_ids = SUITE._protocol_window_ids(
            job.protocol_eval_role, seed=None, count=job.effective_arguments["eval_limit"]
        )
        evaluation_hashes = [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in evaluation_ids]
        run_config["data"]["protocol"] = {
            "consumed": True,
            "manifest_sha256": job.protocol_manifest_sha256,
            "schema_version": SUITE.PROTOCOL_SCHEMA_VERSION,
            "selected_seed": job.protocol_seed,
            "evaluation_role": job.protocol_eval_role,
            "calibration_window_ids": calibration_ids,
            "evaluation_window_ids": evaluation_ids,
            "calibration_token_sha256": "1" * 64,
            "evaluation_token_sha256": "2" * 64,
            "calibration_window_count": len(calibration_ids),
            "evaluation_window_count": len(evaluation_ids),
            "window_token_length": job.effective_arguments["sequence_length"],
            "validation_window_ids": [
                f"validation/fixed/{index:03d}" for index in range(2)
            ],
            "test_window_ids": evaluation_ids,
        }
        run_config["data"]["calib_digest"] = "1" * 64
        run_config["data"]["eval_digest"] = "2" * 64
        run_config["data"]["protocol_activation_sampling"] = {
            "policy": SUITE.PROTOCOL_ACTIVATION_SAMPLING_POLICY,
            "calibration_window_ids": calibration_ids,
            "calibration_window_count": len(calibration_ids),
            "total_token_rows": len(calibration_ids) * job.effective_arguments["sequence_length"],
            "sampled_rows_per_selected_tensor": min(
                job.effective_arguments["selector_activation_sample_rows"],
                len(calibration_ids) * job.effective_arguments["sequence_length"],
            ),
            "all_calibration_windows_traversed": True,
        }
        run_config["protocol_model_binding"] = {
            **model_identity,
            "expected_model_id": job.model_declared,
            "expected_snapshot_commit": revision,
            "model_class": "GPTNeoXForCausalLM",
            "tokenizer_class": "GPTNeoXTokenizerFast",
            "tokenizer_runtime_commit_attestation": (
                "runtime_field_unavailable_asset_sha_bound"
            ),
            "snapshot_files": [
                {
                    "filename": filename,
                    "sha256": digest,
                    "size_bytes": index + 1,
                    "snapshot_commit": revision,
                }
                for index, (filename, digest) in enumerate(
                    SUITE.PROTOCOL_FROZEN_HF_FILE_SHA256.items()
                )
            ],
            "validated": True,
        }
        run_config["data"]["protocol_numerical_path_window_counts"] = {
            "covariance_calibration": 2,
            "activation_risk_calibration": 2,
            "endpoint_nll_evaluation": 3,
            "comfort_recovery_validation": 0,
            "reconstructed_available_unique": 7,
        }
        run_config["protocol_consumer"] = {
            "version": SUITE.PROTOCOL_SCHEMA_VERSION,
            "direct_token_tensor_input": True,
            "text_join_or_retokenization": False,
            "token_repetition": False,
        }
        with (output / "endpoint_window_nll.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "strategy",
                    "window_index",
                    "tokens",
                    "nll",
                    "protocol_window_id",
                    "protocol_window_sha256",
                    "protocol_role",
                    "protocol_seed",
                ],
            )
            writer.writeheader()
            for strategy in ("dense", *job.expected_strategies):
                for index, (window_id, digest) in enumerate(zip(evaluation_ids, evaluation_hashes)):
                    writer.writerow(
                        {
                            "strategy": strategy,
                            "window_index": index,
                            "tokens": job.effective_arguments["sequence_length"] - 1,
                            "nll": 1.0 + index / 100.0,
                            "protocol_window_id": window_id,
                            "protocol_window_sha256": digest,
                            "protocol_role": job.protocol_eval_role,
                            "protocol_seed": None,
                        }
                    )

    artifacts = output / "artifacts"
    artifacts.mkdir()
    reference = artifacts / "reference.bin"
    reference.write_bytes(b"r" * 128)
    strategies = []
    for index, strategy in enumerate(job.expected_strategies):
        artifact = artifacts / f"strategy_{index}.bin"
        artifact.write_bytes(bytes([index + 1]) * 16)
        entry = {
            "strategy": strategy,
            "target_ratio": job.target_rate,
            "artifact_path": artifact.relative_to(output).as_posix(),
            "artifact_sha256": _sha(artifact),
            "artifact_file_bytes": artifact.stat().st_size,
            "artifact_natural_file_bytes": artifact.stat().st_size,
            "artifact_tail_padding_bytes": 0,
        }
        entry.update(
            {
                "reference_artifact_file_bytes": reference.stat().st_size,
                "artifact_to_reference_file_ratio": (
                    artifact.stat().st_size / reference.stat().st_size
                ),
                "artifact_physical_compression_ratio": (
                    reference.stat().st_size / artifact.stat().st_size
                ),
            }
        )
        strategies.append(entry)
    manifest = {
        "production_backend": False,
        "scope": "selected_linear_weights_only",
        "serialized_rate_cap_enforced": True,
        "reference": {
            "path": reference.relative_to(output).as_posix(),
            "sha256": _sha(reference),
            "file_bytes": reference.stat().st_size,
        },
        "strategies": strategies,
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    physical_fields = [
        "strategy",
        "target_ratio",
        "artifact_path",
        "artifact_sha256",
        "artifact_file_bytes",
        "artifact_natural_file_bytes",
        "artifact_tail_padding_bytes",
        "reference_artifact_file_bytes",
        "artifact_to_reference_file_ratio",
        "artifact_physical_compression_ratio",
    ]
    with (output / "artifact_payloads.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=physical_fields)
        writer.writeheader()
        writer.writerows(strategies)
    endpoint_metrics = [
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
    ]
    with (output / "strategy_endpoints.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*physical_fields, *endpoint_metrics])
        writer.writeheader()
        for entry in strategies:
            writer.writerow({**entry, **{field: 0.0 for field in endpoint_metrics}})
    covariance_fields = [
        "layer",
        "collected_diagonal_mean",
        "collected_min_eigenvalue",
        "collected_spectral_scale",
        "configured_damping",
        "configured_damping_ratio",
        "diagonal_shift",
        "diagonal_shift_relative",
        "downstream_covariance_binding",
        "final_min_eigenvalue",
        "final_spectral_scale",
        "float32_storage_floor_rtol",
        "original_min_eigenvalue",
        "original_min_relative",
        "original_spectral_scale",
        "psd_rejection_rtol",
        "spectrum_decomposition_count",
    ]
    with (output / "covariance_psd_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=covariance_fields)
        writer.writeheader()
        for layer in selected:
            writer.writerow(
                {
                    **{field: 0.0 for field in covariance_fields if field != "layer"},
                    "layer": layer,
                    "downstream_covariance_binding": "immutable_prevalidated_input_covariance",
                    "spectrum_decomposition_count": 1,
                }
            )
    (output / "COMPLETED").write_text('{"completed": true}\n', encoding="utf-8")


def test_committed_matrix_expands_model_seed_scope_rate_jobs_without_gpu() -> None:
    definition = SUITE.load_suite_definition(
        REPO_ROOT / "configs" / "large_scale_hessian_suite_20260714.json",
        repo_root=REPO_ROOT,
    )
    jobs = SUITE.expand_jobs(definition, environment={})
    assert len(jobs) == 69
    assert sum(job.model_availability == "required" for job in jobs) == 24
    assert sum(job.model_availability == "optional" for job in jobs) == 45
    assert len({job.job_id for job in jobs}) == len(jobs)

    pythia70 = [job for job in jobs if job.stage_id == "pythia70m_full_mlp_scalability"]
    assert len(pythia70) == 8 * 3
    assert {job.seed for job in pythia70} == {17, 29, 43, 59, 71, 89, 101, 113}
    assert {job.target_rate for job in pythia70} == {0.258, 0.275, 0.3}
    assert all(job.tensor_scope["layers"] == [0, 1, 2, 3, 4, 5] for job in pythia70)
    assert all(job.tensor_scope["expected_selected_tensors"] == 12 for job in pythia70)
    assert all(job.evidence_role == "scalability_smoke" for job in jobs)
    assert all(job.protocol_manifest_consumed is False for job in jobs)
    assert all(job.seed_aggregation_allowed is False for job in jobs)
    assert all(job.effective_arguments["target_ratios"] == [job.target_rate] for job in jobs)
    assert all(job.effective_arguments["endpoint_target"] == job.target_rate for job in jobs)
    scopes = {
        job.stage_id: (job.model_declared, tuple(job.tensor_scope["module_types"]))
        for job in jobs
    }
    assert scopes["opt125m_architecture_control"] == ("facebook/opt-125m", ("fc1", "fc2"))
    assert scopes["qwen3_06b_architecture_control"] == (
        "Qwen/Qwen3-0.6B",
        ("up_proj", "down_proj"),
    )
    assert scopes["llama32_1b_architecture_control"] == (
        "meta-llama/Llama-3.2-1B",
        ("up_proj", "down_proj"),
    )


def test_stage_runner_overrides_define_auditable_method_variants(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("# fixture\n", encoding="utf-8")
    path = _minimal_config(tmp_path, runner)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["common"].update(
        {
            "s_method": "wanda",
            "l_method": "whitened_svd",
            "residual_order": "s_then_l",
            "covariance_mode": "full",
            "rate_allocation": "local_guard",
        }
    )
    payload["stages"][0]["runner_overrides"] = {
        "s_method": "magnitude",
        "l_method": "svd",
        "residual_order": "l_then_s",
        "covariance_mode": "diagonal",
        "rate_allocation": "global_exact",
        "strict_sparse_refit": "obs",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    definition = SUITE.load_suite_definition(path, repo_root=tmp_path)
    job = SUITE.expand_jobs(definition, environment={})[0]
    assert job.effective_arguments["s_method"] == "magnitude"
    assert job.effective_arguments["l_method"] == "svd"
    assert job.effective_arguments["residual_order"] == "l_then_s"
    assert job.effective_arguments["covariance_mode"] == "diagonal"
    assert job.effective_arguments["rate_allocation"] == "global_exact"
    assert job.effective_arguments["strict_sparse_refit"] == "obs"
    command = SUITE.build_runner_command(
        job, python_executable="python", runner=definition.runner
    )
    joined = " ".join(command)
    assert "--strict-sparse-refit obs" in joined
    assert "--rate-allocation global_exact" in joined

    payload["stages"][0]["runner_overrides"] = {"model": "forbidden"}
    with pytest.raises(SUITE.SuiteConfigError, match="unsupported arguments"):
        SUITE.validate_suite_payload(payload)


def test_confirmatory_frontier_expands_24_protocol_bound_jobs() -> None:
    definition = SUITE.load_suite_definition(
        REPO_ROOT / "configs" / "confirmatory_hessian_pythia70_frontier_v2_20260715.json",
        repo_root=REPO_ROOT,
    )
    jobs = SUITE.expand_jobs(definition, environment={})
    assert len(jobs) == 8 * 3
    assert {job.seed for job in jobs} == {17, 29, 43, 59, 71, 89, 101, 113}
    assert {job.target_rate for job in jobs} == {0.258, 0.275, 0.3}
    assert all(job.evidence_role == "confirmatory" for job in jobs)
    assert all(job.protocol_manifest_consumed is True for job in jobs)
    assert all(job.seed_aggregation_allowed is True for job in jobs)
    assert all(job.protocol_seed == job.seed for job in jobs)
    assert all(job.protocol_eval_role == "test" for job in jobs)
    assert all(
        job.protocol_manifest_sha256
        == "9e8315ad6bb60a7a0c17765058564f23f6e5eff2337e089f41a451eb51e15547"
        for job in jobs
    )
    assert set(jobs[0].numerical_source_snapshot) == {
        "runner",
        "protocol_consumer",
        "legacy_runner",
        "codec",
        "hessian_repair",
        "base_runner",
        "model_data",
        "protocol_manifest",
    }
    assert (
        jobs[0].numerical_source_snapshot["protocol_manifest"]["sha256"]
        == jobs[0].protocol_manifest_sha256
    )
    assert "comfort_sweep.csv" not in definition.expected_outputs
    assert "comfort_summary.csv" not in definition.expected_outputs
    assert not any(path.startswith("figures/") for path in definition.expected_outputs)
    assert definition.raw["analysis_plan"]["primary_contrast"] == {
        "candidate": "Q+S+L_QL_budget",
        "reference": "Q+L",
        "design_label": "strict_same_byte_composite_vs_ql",
        "directional_hypothesis": "candidate_lower",
        "rate_match_rule": SUITE.PRIMARY_RATE_MATCH_RULE,
    }
    assert definition.raw["analysis_plan"]["replicate_unit"] == "seed"
    assert definition.raw["analysis_plan"]["fixed_test_window_role"] == (
        "paired_diagnostic_only_not_independent_replicates"
    )
    assert definition.raw["analysis_plan"]["within_seed_rate_aggregation"] == (
        "unweighted_arithmetic_mean_of_candidate_minus_reference_heldout_nll_at_frozen_rates_0p258_0p275_0p300"
    )
    assert definition.raw["analysis_plan"]["p_value"] == "count(T_perm <= T_obs)/256"
    command = SUITE.build_runner_command(
        jobs[0], python_executable="python", runner=definition.runner
    )
    joined = " ".join(command)
    assert "--protocol-manifest results/confirmatory_hessian_protocol_20260714/protocol.json" in joined
    assert "--protocol-manifest-sha256 9e8315ad6bb60a7a0c17765058564f23f6e5eff2337e089f41a451eb51e15547" in joined
    assert "--protocol-seed 17" in joined
    assert "--protocol-eval-role test" in joined
    assert "--skip-comfort" in command
    assert "--skip-plots" in command


def test_confirmatory_analysis_plan_contrasts_are_fail_closed() -> None:
    path = REPO_ROOT / "configs" / "confirmatory_hessian_pythia70_frontier_v2_20260715.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["analysis_plan"]["primary_contrast"]["reference"] = "Q"
    with pytest.raises(SUITE.SuiteConfigError, match="primary contrast"):
        SUITE.validate_suite_payload(payload)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["analysis_plan"]["randomization"] = "unspecified"
    with pytest.raises(SUITE.SuiteConfigError, match="analysis_plan.randomization"):
        SUITE.validate_suite_payload(payload)


def test_signed_pilot_source_and_job_hashes_are_backward_compatible() -> None:
    definition = SUITE.load_suite_definition(
        REPO_ROOT / "configs" / "large_scale_hessian_pilot_20260714.json",
        repo_root=REPO_ROOT,
    )
    recorded = SUITE.load_recorded_model_arguments(definition)
    jobs = SUITE.expand_jobs(
        definition,
        environment={},
        recorded_model_arguments=recorded,
    )
    persisted = json.loads(
        (definition.output_root / "suite_manifest.json").read_text(encoding="utf-8")
    )
    expected_job_hashes = {
        entry["job_id"]: entry["job_config_sha256"] for entry in persisted["jobs"]
    }
    assert {job.numerical_source_sha256 for job in jobs} == {
        "d5444b26c3f77718ba611ef95c386c28799ca511cc9be50ade2628467b6266b3"
    }
    assert {job.job_id: job.job_config_sha256 for job in jobs} == expected_job_hashes
    assert all(job.protocol_manifest is None for job in jobs)
    assert all("protocol_manifest" not in job.effective_arguments for job in jobs)


def test_large_model_method_matrix_exposes_budget_bands_and_allocator_control() -> None:
    definition = SUITE.load_suite_definition(
        REPO_ROOT / "configs" / "large_model_method_ablation_20260716.json",
        repo_root=REPO_ROOT,
    )
    jobs = SUITE.expand_jobs(definition, environment={})
    assert len(jobs) == 15
    assert len({job.job_id for job in jobs}) == 15
    assert all(job.seed == 17 and job.target_rate == pytest.approx(0.258) for job in jobs)
    primary = next(job for job in jobs if job.stage_id == "pythia70m_full_mlp_global_obs")
    guard = next(job for job in jobs if job.stage_id == "pythia70m_full_mlp_local_guard_obs")
    full_svd = next(
        job for job in jobs if job.stage_id == "pythia70m_full_mlp_global_obs_full_svd"
    )
    assert primary.effective_arguments["rate_allocation"] == "global_exact"
    assert primary.effective_arguments["global_frontier_budget_multipliers"] == [1.25, 1.5, 2.0]
    assert primary.effective_arguments["lowrank_svd_solver"] == "randomized"
    assert primary.effective_arguments["lowrank_svd_oversampling"] == 4
    assert primary.effective_arguments["lowrank_svd_niter"] == 2
    assert guard.effective_arguments["rate_allocation"] == "local_guard"
    assert full_svd.effective_arguments["lowrank_svd_solver"] == "full"
    assert primary.effective_arguments["comfort_epsilons"][:3] == [0.0, 0.0625, 0.125]
    command = SUITE.build_runner_command(
        primary,
        python_executable="python",
        runner=definition.runner,
    )
    joined = " ".join(command)
    assert "--global-frontier-budget-multipliers 1.25,1.5,2.0" in joined
    assert "--lowrank-svd-solver randomized" in joined


def test_large_model_v2_is_staged_rank_complete_and_explicitly_omits_block_scale() -> None:
    definition = SUITE.load_suite_definition(
        REPO_ROOT / "configs" / "large_model_method_ablation_v2_20260716.json",
        repo_root=REPO_ROOT,
    )
    jobs = SUITE.expand_jobs(definition, environment={})
    assert len(jobs) == 28
    assert len({job.job_id for job in jobs}) == 28
    assert "Q_block_scale" not in definition.expected_strategies
    assert all(job.effective_arguments["skip_block_scale"] is True for job in jobs)
    assert all(job.effective_arguments["calib_limit"] == 32 for job in jobs)
    assert all(job.effective_arguments["sequence_length"] == 256 for job in jobs)
    assert all(job.effective_arguments["global_frontier_top_ranks"] == 3 for job in jobs)
    assert all(
        job.effective_arguments["allocation_rank_grid"]
        == [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        for job in jobs
    )
    sentinels = [job for job in jobs if job.stage_id.startswith("s0_")]
    assert len(sentinels) == 4
    assert all(job.tensor_scope["expected_selected_tensors"] == 1 for job in sentinels)
    assert all(job.effective_arguments["eval_limit"] == 8 for job in sentinels)
    qwen_rates = [job for job in jobs if job.stage_id == "s1_qwen_primary_rates"]
    assert {job.target_rate for job in qwen_rates} == {0.258, 0.275, 0.3}
    command = SUITE.build_runner_command(
        sentinels[0], python_executable="python", runner=definition.runner
    )
    joined = " ".join(command)
    assert "--skip-block-scale" in joined
    assert "--allocation-rank-grid 0,1,2,4,8,12,16,24,32,48,64,96,128" in joined


def test_large_model_v3_adds_nested_nojoint_control_and_19_bounded_jobs() -> None:
    definition = SUITE.load_suite_definition(
        REPO_ROOT / "configs" / "large_model_global_controls_v3_20260716.json",
        repo_root=REPO_ROOT,
    )
    jobs = SUITE.expand_jobs(definition, environment={})
    assert len(jobs) == 19
    assert len({job.job_id for job in jobs}) == 19
    assert "Q+S_OBS_or_L_global" in definition.expected_strategies
    assert "Q_block_scale" not in definition.expected_strategies
    assert all(
        job.effective_arguments["include_global_single_component_controls"] is True
        for job in jobs
    )
    assert all(job.effective_arguments["rate_allocation"] == "global_exact" for job in jobs)
    assert all(job.effective_arguments["skip_block_scale"] is True for job in jobs)
    assert all(job.effective_arguments["calib_limit"] == 32 for job in jobs)
    assert all(job.effective_arguments["sequence_length"] == 256 for job in jobs)
    assert definition.raw["resource_policy"]["enforce_at_runtime"] is True
    assert all(job.resource_policy["enforce_at_runtime"] is True for job in jobs)
    assert all(job.effective_arguments["model_snapshot_manifest"] for job in jobs)
    assert all(
        len(job.effective_arguments["model_snapshot_manifest_sha256"]) == 64
        and len(job.effective_arguments["model_snapshot_aggregate_sha256"]) == 64
        for job in jobs
    )
    sentinels = [job for job in jobs if job.stage_id.startswith("s0_")]
    assert len(sentinels) == 3
    assert all(job.tensor_scope["expected_selected_tensors"] == 1 for job in sentinels)
    assert all(job.effective_arguments["eval_limit"] == 8 for job in sentinels)
    primary = [job for job in jobs if job.stage_id == "s1_qwen_three_depth_rates"]
    assert {job.target_rate for job in primary} == {0.258, 0.275, 0.3}
    command = SUITE.build_runner_command(
        sentinels[0], python_executable="python", runner=definition.runner
    )
    assert "--include-global-single-component-controls" in command
    assert "--model-snapshot-manifest" in command
    assert "--resource-gate-manifest" in command
    qwen_sentinel = next(
        job for job in sentinels if job.stage_id == "s0_qwen_global_control_sentinel"
    )
    assert qwen_sentinel.tensor_scope["endpoint_identity_pairs"] == [
        ["Q+L", "Q+L_global"]
    ]


def test_model_snapshot_repository_root_is_independent_of_process_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    definition = SUITE.load_suite_definition(
        REPO_ROOT / "configs" / "large_model_global_controls_v3_20260716.json",
        repo_root=REPO_ROOT,
    )
    job = SUITE.expand_jobs(definition, environment={})[0]
    monkeypatch.chdir(tmp_path)
    assert SUITE._job_repo_root(job) == REPO_ROOT.resolve()


def test_pilot_matrix_is_single_seed_single_rate_and_keeps_all_native_strategies() -> None:
    definition = SUITE.load_suite_definition(
        REPO_ROOT / "configs" / "large_scale_hessian_pilot_20260714.json",
        repo_root=REPO_ROOT,
    )
    jobs = SUITE.expand_jobs(definition, environment={})
    assert len(jobs) == 3
    assert {job.stage_id for job in jobs} == {
        "pythia70m_full_mlp_pilot",
        "opt125m_depth_mlp_pilot",
        "qwen3_06b_depth_mlp_pilot",
    }
    assert {job.seed for job in jobs} == {17}
    assert {job.target_rate for job in jobs} == {0.258}
    assert all(job.evidence_role == "scalability_smoke" for job in jobs)
    assert all(job.seed_aggregation_allowed is False for job in jobs)
    assert all(job.expected_strategies == definition.expected_strategies for job in jobs)


def test_persisted_model_locator_makes_optional_check_portable_and_tamper_evident(
    tmp_path: Path,
) -> None:
    definition, _ = _make_fake_suite(tmp_path)
    definition.stages[0]["model_availability"] = "optional"
    recorded_job = SUITE.expand_jobs(
        definition, environment={"UNIT_MODEL": "/server/cache/model/snapshot/abc123"}
    )[0]
    definition.output_root.mkdir(parents=True)
    manifest_path = definition.output_root / "suite_manifest.json"
    manifest = {
        "schema_version": SUITE.MANIFEST_SCHEMA_VERSION,
        "suite_id": definition.suite_id,
        "suite_config_sha256": definition.config_sha256,
        "numerical_source_snapshot": SUITE.collect_numerical_source_snapshot(definition),
        "jobs": [
            {
                "job_id": recorded_job.job_id,
                "stage_id": recorded_job.stage_id,
                "model_argument": recorded_job.model_argument,
                "job_config_sha256": recorded_job.job_config_sha256,
                "numerical_source_sha256": recorded_job.numerical_source_sha256,
                "status": "planned",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (definition.output_root / "suite_summary.csv").write_text("fixture\n", encoding="utf-8")
    (definition.output_root / "suite_summary.md").write_text("fixture\n", encoding="utf-8")

    recorded = SUITE.load_recorded_model_arguments(definition)
    audited_job = SUITE.expand_jobs(
        definition, environment={}, recorded_model_arguments=recorded
    )[0]
    assert audited_job.model_argument == "/server/cache/model/snapshot/abc123"
    assert audited_job.job_config_sha256 == recorded_job.job_config_sha256
    selected, skipped = SUITE.select_jobs(
        [audited_job], job_id=None, stages=[], include_optional=True
    )
    assert selected == [audited_job]
    assert skipped == []
    assert SUITE.check_suite(definition, [audited_job], selected)["ok"] is True

    manifest["jobs"][0]["model_argument"] = "/tampered/snapshot"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tampered = SUITE.load_recorded_model_arguments(definition)
    tampered_job = SUITE.expand_jobs(
        definition, environment={}, recorded_model_arguments=tampered
    )[0]
    report = SUITE.check_suite(definition, [tampered_job], [tampered_job])
    assert report["ok"] is False
    assert any("persisted job hash differs" in item for item in report["persisted_errors"])


def test_numerical_subprocess_inherits_repo_src_on_pythonpath(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "src").mkdir()
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(SUITE.subprocess, "run", fake_run)
    SUITE._run_process(
        ["python", "runner.py"],
        cwd=tmp_path,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str((tmp_path / "src").resolve())


def test_confirmatory_or_seed_aggregation_claims_are_rejected_without_protocol_consumption(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("# fixture\n", encoding="utf-8")
    config_path = _minimal_config(tmp_path, runner)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["stages"][0]["evidence_role"] = "confirmatory"
    with pytest.raises(SUITE.SuiteConfigError, match="cannot be confirmatory"):
        SUITE.validate_suite_payload(payload)

    payload["stages"][0]["evidence_role"] = "scalability_smoke"
    payload["stages"][0]["seed_aggregation_allowed"] = True
    with pytest.raises(SUITE.SuiteConfigError, match="cannot aggregate seeds"):
        SUITE.validate_suite_payload(payload)

    payload["stages"][0]["seed_aggregation_allowed"] = False
    payload["stages"][0]["protocol_manifest_consumed"] = True
    with pytest.raises(SUITE.SuiteConfigError, match="current runner cannot consume"):
        SUITE.validate_suite_payload(payload)


def test_protocol_config_fails_closed_on_unsafe_path_or_manifest_drift(tmp_path: Path) -> None:
    config_path = _minimal_protocol_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["stages"][0]["protocol_manifest"] = "../protocol.json"
    with pytest.raises(SUITE.SuiteConfigError, match="safe repository-relative"):
        SUITE.validate_suite_payload(payload)

    manifest_path = tmp_path / "protocol" / "protocol.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SUITE.SuiteConfigError, match="SHA-256 differs"):
        SUITE.load_suite_definition(config_path, repo_root=tmp_path)


def test_protocol_config_prohibits_model_override_and_requires_manifest_model_binding(
    tmp_path: Path,
) -> None:
    config_path = _minimal_protocol_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["stages"][0]["model_override_env"] = "UNIT_MODEL"
    with pytest.raises(SUITE.SuiteConfigError, match="prohibited"):
        SUITE.validate_suite_payload(payload)

    payload["stages"][0].pop("model_override_env")
    payload["stages"][0]["model"] = "wrong/model"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SUITE.SuiteConfigError, match="stage model differs"):
        SUITE.load_suite_definition(config_path, repo_root=tmp_path)

    payload["stages"][0]["model"] = "local/tiny"
    payload["stages"][0]["revision"] = "wrong-revision"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SUITE.SuiteConfigError, match="stage revision differs"):
        SUITE.load_suite_definition(config_path, repo_root=tmp_path)


def test_protocol_output_proof_and_ordered_endpoint_windows_are_required(tmp_path: Path) -> None:
    config_path = _minimal_protocol_config(tmp_path)
    definition = SUITE.load_suite_definition(config_path, repo_root=tmp_path)
    job = SUITE.expand_jobs(definition, environment={})[0]
    _write_fake_completed_output(job)
    evidence = SUITE.validate_runner_outputs(job, require_suite_record=False)
    assert evidence["protocol"]["consumed"] is True
    assert len(evidence["protocol_evaluation_window_sha256"]) == 3

    run_path = job.output_dir / "run_config.json"
    run_config = json.loads(run_path.read_text(encoding="utf-8"))
    run_config["data"]["protocol"]["selected_seed"] = 29
    run_path.write_text(json.dumps(run_config), encoding="utf-8")
    with pytest.raises(SUITE.EvidenceError, match="selected_seed differs"):
        SUITE.validate_runner_outputs(job, require_suite_record=False)


def test_protocol_endpoint_window_tampering_is_rejected(tmp_path: Path) -> None:
    config_path = _minimal_protocol_config(tmp_path)
    definition = SUITE.load_suite_definition(config_path, repo_root=tmp_path)
    job = SUITE.expand_jobs(definition, environment={})[0]
    _write_fake_completed_output(job)
    path = job.output_dir / "endpoint_window_nll.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["protocol_window_id"], rows[1]["protocol_window_id"] = (
        rows[1]["protocol_window_id"],
        rows[0]["protocol_window_id"],
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SUITE.EvidenceError, match="window order differs"):
        SUITE.validate_runner_outputs(job, require_suite_record=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_role", "validation", "protocol role differs"),
        ("protocol_seed", "17", "fixed evaluation seed differs"),
    ],
)
def test_protocol_endpoint_role_and_fixed_seed_are_required(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    definition = SUITE.load_suite_definition(
        _minimal_protocol_config(tmp_path), repo_root=tmp_path
    )
    job = SUITE.expand_jobs(definition, environment={})[0]
    _write_fake_completed_output(job)
    path = job.output_dir / "endpoint_window_nll.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0][field] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SUITE.EvidenceError, match=message):
        SUITE.validate_runner_outputs(job, require_suite_record=False)


def test_protocol_activation_sampling_must_cover_all_calibration_windows(tmp_path: Path) -> None:
    definition = SUITE.load_suite_definition(
        _minimal_protocol_config(tmp_path), repo_root=tmp_path
    )
    job = SUITE.expand_jobs(definition, environment={})[0]
    _write_fake_completed_output(job)
    path = job.output_dir / "run_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["data"]["protocol_activation_sampling"][
        "all_calibration_windows_traversed"
    ] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SUITE.EvidenceError, match="all_calibration_windows_traversed differs"):
        SUITE.validate_runner_outputs(job, require_suite_record=False)


def test_protocol_physical_tables_are_recomputed_from_artifact_files(tmp_path: Path) -> None:
    definition = SUITE.load_suite_definition(
        _minimal_protocol_config(tmp_path), repo_root=tmp_path
    )
    job = SUITE.expand_jobs(definition, environment={})[0]
    _write_fake_completed_output(job)
    path = job.output_dir / "artifact_payloads.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["artifact_to_reference_file_ratio"] = "0.2"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SUITE.EvidenceError, match="serialized artifact ratio is inconsistent"):
        SUITE.validate_runner_outputs(job, require_suite_record=False)


def test_protocol_actual_serialized_rate_must_stay_within_declared_tolerance(
    tmp_path: Path,
) -> None:
    config_path = _minimal_protocol_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["stages"][0]["rates"] = [0.1]
    payload["common"]["rate_tolerance"] = 0.01
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    definition = SUITE.load_suite_definition(config_path, repo_root=tmp_path)
    job = SUITE.expand_jobs(definition, environment={})[0]
    _write_fake_completed_output(job)
    with pytest.raises(SUITE.EvidenceError, match=r"target\+tolerance"):
        SUITE.validate_runner_outputs(job, require_suite_record=False)


def test_scalability_physical_tables_are_also_recomputed(tmp_path: Path) -> None:
    definition, job = _make_fake_suite(tmp_path)
    del definition
    _write_fake_completed_output(job)
    path = job.output_dir / "artifact_payloads.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["artifact_to_reference_file_ratio"] = "0.2"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SUITE.EvidenceError, match="serialized artifact ratio is inconsistent"):
        SUITE.validate_runner_outputs(job, require_suite_record=False)


def test_global_exact_q_l_natural_bytes_are_bound_to_the_actual_uncapped_file(
    tmp_path: Path,
) -> None:
    definition, original_job = _make_fake_suite(tmp_path)
    del definition
    effective = {**original_job.effective_arguments, "rate_allocation": "global_exact"}
    job = replace(original_job, effective_arguments=effective)
    _write_fake_completed_output(job)
    run_path = job.output_dir / "run_config.json"
    run_config = json.loads(run_path.read_text(encoding="utf-8"))
    run_config["arguments"]["rate_allocation"] = "global_exact"
    run_path.write_text(json.dumps(run_config), encoding="utf-8")

    manifest_path = job.output_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ql_manifest = next(
        row for row in manifest["strategies"] if row["strategy"] == "Q+L"
    )
    ql_manifest["artifact_natural_file_bytes"] = 15
    ql_manifest["artifact_tail_padding_bytes"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for relative in ("artifact_payloads.csv", "strategy_endpoints.csv"):
        path = job.output_dir / relative
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        ql_row = next(row for row in rows if row["strategy"] == "Q+L")
        ql_row["artifact_natural_file_bytes"] = "15"
        ql_row["artifact_tail_padding_bytes"] = "1"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    with pytest.raises(SUITE.EvidenceError, match="uncapped reference strategy"):
        SUITE.validate_runner_outputs(job, require_suite_record=False)


def test_scalability_exact_token_counts_are_required(tmp_path: Path) -> None:
    definition, job = _make_fake_suite(tmp_path)
    del definition
    _write_fake_completed_output(job)
    path = job.output_dir / "run_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["actual_eval_tokens"] -= 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SUITE.EvidenceError, match="eval token count differs"):
        SUITE.validate_runner_outputs(job, require_suite_record=False)


def test_scalability_nonfinite_endpoint_metric_is_rejected(tmp_path: Path) -> None:
    definition, job = _make_fake_suite(tmp_path)
    del definition
    _write_fake_completed_output(job)
    path = job.output_dir / "strategy_endpoints.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["hessian_cost"] = "nan"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SUITE.EvidenceError, match="non-finite hessian_cost"):
        SUITE.validate_runner_outputs(job, require_suite_record=False)


def test_global_exact_allocator_fallback_and_open_byte_ledger_are_rejected() -> None:
    allocator = {
        "mode": "global_exact",
        "selection_source": "global_exact_canonical_layout_pareto_frontier",
        "strict_file_byte_feasible": True,
        "frontier_coarsening_events": 0,
        "q_l_cap_natural_file_bytes": 1024,
        "selected_qsl_natural_file_bytes": 960,
        "unused_natural_bytes_before_tail_padding": 64,
        "full_serializer_cross_checks": 2,
        "selected_hessian_cost": 1.0,
    }
    SUITE._validate_global_allocator(
        {"rate_allocator": allocator}, {"rate_allocation": "global_exact"}
    )

    fallback = {**allocator, "selection_source": "fallback_local_guard"}
    with pytest.raises(SUITE.EvidenceError, match="selection_source differs"):
        SUITE._validate_global_allocator(
            {"rate_allocator": fallback}, {"rate_allocation": "global_exact"}
        )
    open_ledger = {**allocator, "unused_natural_bytes_before_tail_padding": 0}
    with pytest.raises(SUITE.EvidenceError, match="byte ledger does not close"):
        SUITE._validate_global_allocator(
            {"rate_allocator": open_ledger}, {"rate_allocation": "global_exact"}
        )


def test_global_control_reports_and_nojoint_dominance_are_fail_closed() -> None:
    cap = 1024

    def report(strategy: str, cost: float) -> dict[str, object]:
        return {
            "mode": "global_exact",
            "endpoint_label": strategy,
            "selection_source": "global_exact_canonical_layout_pareto_frontier",
            "strict_file_byte_feasible": True,
            "frontier_coarsening_events": 0,
            "fallback_policy": "forbidden_fail_closed",
            "q_l_cap_natural_file_bytes": cap,
            "selected_natural_file_bytes": 960,
            "unused_natural_bytes_before_tail_padding": 64,
            "full_serializer_cross_checks": 2,
            "selected_hessian_cost": cost,
        }

    controls = {
        "Q+S_OBS_global": report("Q+S_OBS_global", 1.2),
        "Q+L_global": report("Q+L_global", 1.0),
        "Q+S_OBS_or_L_global": report("Q+S_OBS_or_L_global", 1.0),
    }
    allocator = {
        "mode": "global_exact",
        "selection_source": "global_exact_canonical_layout_pareto_frontier",
        "strict_file_byte_feasible": True,
        "frontier_coarsening_events": 0,
        "q_l_cap_natural_file_bytes": cap,
        "selected_qsl_natural_file_bytes": 944,
        "unused_natural_bytes_before_tail_padding": 80,
        "full_serializer_cross_checks": 2,
        "selected_hessian_cost": 0.9,
        "global_control_reports": controls,
        "joint_control_strategy": "Q+S_OBS_or_L_global",
        "joint_control_weakly_dominated": True,
        "joint_candidate_incremental_hessian_gain": 0.1,
        "nonjoint_union_weakly_dominates_pure_controls": True,
        "best_pure_control_hessian_cost": 1.0,
        "nonjoint_heterogeneous_gain_over_best_pure": 0.0,
    }
    arguments = {
        "rate_allocation": "global_exact",
        "include_global_single_component_controls": True,
    }
    SUITE._validate_global_allocator({"rate_allocator": allocator}, arguments)

    missing = {**allocator, "global_control_reports": {k: v for k, v in controls.items() if k != "Q+S_OBS_or_L_global"}}
    with pytest.raises(SUITE.EvidenceError, match="report set differs"):
        SUITE._validate_global_allocator({"rate_allocator": missing}, arguments)

    inconsistent = {**allocator, "joint_candidate_incremental_hessian_gain": 0.0}
    with pytest.raises(SUITE.EvidenceError, match="incremental Hessian gain is inconsistent"):
        SUITE._validate_global_allocator({"rate_allocator": inconsistent}, arguments)


def test_multilayer_nojoint_union_may_strictly_beat_each_pure_family() -> None:
    cap = 2048

    def control(strategy: str, cost: float) -> dict[str, object]:
        return {
            "mode": "global_exact",
            "endpoint_label": strategy,
            "selection_source": "global_exact_canonical_layout_pareto_frontier",
            "strict_file_byte_feasible": True,
            "frontier_coarsening_events": 0,
            "fallback_policy": "forbidden_fail_closed",
            "q_l_cap_natural_file_bytes": cap,
            "selected_natural_file_bytes": 1900,
            "unused_natural_bytes_before_tail_padding": 148,
            "full_serializer_cross_checks": 2,
            "selected_hessian_cost": cost,
        }

    controls = {
        "Q+S_OBS_global": control("Q+S_OBS_global", 1.1),
        "Q+L_global": control("Q+L_global", 1.2),
        "Q+S_OBS_or_L_global": control("Q+S_OBS_or_L_global", 0.8),
    }
    allocator = {
        "mode": "global_exact",
        "selection_source": "global_exact_canonical_layout_pareto_frontier",
        "strict_file_byte_feasible": True,
        "frontier_coarsening_events": 0,
        "q_l_cap_natural_file_bytes": cap,
        "selected_qsl_natural_file_bytes": 1880,
        "unused_natural_bytes_before_tail_padding": 168,
        "full_serializer_cross_checks": 2,
        "selected_hessian_cost": 0.7,
        "global_control_reports": controls,
        "joint_control_strategy": "Q+S_OBS_or_L_global",
        "joint_control_weakly_dominated": True,
        "joint_candidate_incremental_hessian_gain": 0.1,
        "nonjoint_union_weakly_dominates_pure_controls": True,
        "best_pure_control_hessian_cost": 1.1,
        "nonjoint_heterogeneous_gain_over_best_pure": 0.3,
    }
    arguments = {
        "rate_allocation": "global_exact",
        "include_global_single_component_controls": True,
    }
    SUITE._validate_global_allocator(
        {"rate_allocator": allocator, "selected_layers": ["a", "b"]}, arguments
    )
    with pytest.raises(SUITE.EvidenceError, match="one-layer no-joint union differs"):
        SUITE._validate_global_allocator(
            {"rate_allocator": allocator, "selected_layers": ["a"]}, arguments
        )


def test_global_allocator_is_bound_to_endpoint_costs_and_artifact_natural_bytes() -> None:
    cap = 1024

    def control(strategy: str, natural: int, cost: float) -> dict[str, object]:
        return {
            "mode": "global_exact",
            "endpoint_label": strategy,
            "selection_source": "global_exact_canonical_layout_pareto_frontier",
            "strict_file_byte_feasible": True,
            "frontier_coarsening_events": 0,
            "fallback_policy": "forbidden_fail_closed",
            "q_l_cap_natural_file_bytes": cap,
            "selected_natural_file_bytes": natural,
            "unused_natural_bytes_before_tail_padding": cap - natural,
            "full_serializer_cross_checks": 2,
            "selected_hessian_cost": cost,
        }

    controls = {
        "Q+S_OBS_global": control("Q+S_OBS_global", 930, 1.2),
        "Q+L_global": control("Q+L_global", 940, 1.0),
        "Q+S_OBS_or_L_global": control("Q+S_OBS_or_L_global", 945, 1.0),
    }
    allocator = {
        "mode": "global_exact",
        "selection_source": "global_exact_canonical_layout_pareto_frontier",
        "strict_file_byte_feasible": True,
        "frontier_coarsening_events": 0,
        "q_l_cap_natural_file_bytes": cap,
        "selected_natural_file_bytes": 944,
        "selected_qsl_natural_file_bytes": 944,
        "unused_natural_bytes_before_tail_padding": 80,
        "full_serializer_cross_checks": 2,
        "selected_hessian_cost": 0.9,
        "global_control_reports": controls,
        "joint_control_strategy": "Q+S_OBS_or_L_global",
        "joint_control_weakly_dominated": True,
        "joint_candidate_incremental_hessian_gain": 0.1,
        "nonjoint_union_weakly_dominates_pure_controls": True,
        "best_pure_control_hessian_cost": 1.0,
        "nonjoint_heterogeneous_gain_over_best_pure": 0.0,
    }
    endpoint_rows = {
        strategy: {"hessian_cost": cost}
        for strategy, cost in {
            "Q+S+L_QL_budget": 0.9,
            "Q+S_OBS_global": 1.2,
            "Q+L_global": 1.0,
            "Q+S_OBS_or_L_global": 1.0,
        }.items()
    }
    artifacts = {
        "Q+L": {"natural_file_bytes": cap},
        "Q+S+L_QL_budget": {"natural_file_bytes": 944},
        "Q+S_OBS_global": {"natural_file_bytes": 930},
        "Q+L_global": {"natural_file_bytes": 940},
        "Q+S_OBS_or_L_global": {"natural_file_bytes": 945},
    }
    arguments = {
        "rate_allocation": "global_exact",
        "include_global_single_component_controls": True,
    }
    run_config = {"rate_allocator": allocator, "selected_layers": ["a"]}
    SUITE._validate_global_allocator(
        run_config,
        arguments,
        endpoint_rows=endpoint_rows,
        audited_artifacts=artifacts,
    )

    bad_costs = {key: dict(value) for key, value in endpoint_rows.items()}
    bad_costs["Q+L_global"]["hessian_cost"] = 9.0
    with pytest.raises(SUITE.EvidenceError, match="Hessian cost differs"):
        SUITE._validate_global_allocator(
            run_config,
            arguments,
            endpoint_rows=bad_costs,
            audited_artifacts=artifacts,
        )

    bad_artifacts = {key: dict(value) for key, value in artifacts.items()}
    bad_artifacts["Q+S+L_QL_budget"]["natural_file_bytes"] = 943
    with pytest.raises(SUITE.EvidenceError, match="selected natural bytes differ"):
        SUITE._validate_global_allocator(
            run_config,
            arguments,
            endpoint_rows=endpoint_rows,
            audited_artifacts=bad_artifacts,
        )


def test_two_stage_suite_contract_accepts_disjoint_splits_and_skipped_comfort(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("# fixture\n", encoding="utf-8")
    config_path = _minimal_config(tmp_path, runner)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["evidence_contract"]["current_data_window_policy"] = (
        SUITE.TWO_STAGE_WINDOW_POLICY
    )
    payload["expected_outputs"].extend(
        [
            "allocation_validation_rerank.csv",
            "allocation_validation_window_nll.csv",
            "endpoint_window_nll.csv",
        ]
    )
    payload["common"].update(
        {
            "calibration_split": "train",
            "selection_split": "validation",
            "test_split": "test",
            "selection_limit": 2,
            "rate_allocation": "global_exact",
            "include_global_single_component_controls": True,
            "two_stage_selection": True,
            "selection_top_k": 2,
            "skip_comfort": True,
        }
    )
    payload["stages"][0]["data_window_independence"] = (
        SUITE.TWO_STAGE_WINDOW_INDEPENDENCE
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    definition = SUITE.load_suite_definition(config_path, repo_root=tmp_path)
    assert definition.common["two_stage_selection"] is True

    payload["stages"][0]["data_window_independence"] = (
        SUITE.SHARED_SEQUENTIAL_INDEPENDENCE
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SUITE.SuiteConfigError, match="active split/window policy"):
        SUITE.load_suite_definition(config_path, repo_root=tmp_path)


def test_two_stage_stage_overrides_cannot_break_the_protocol(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("# fixture\n", encoding="utf-8")
    config_path = _minimal_config(tmp_path, runner)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["evidence_contract"]["current_data_window_policy"] = (
        SUITE.TWO_STAGE_WINDOW_POLICY
    )
    payload["expected_outputs"].extend(
        [
            "allocation_validation_rerank.csv",
            "allocation_validation_window_nll.csv",
            "endpoint_window_nll.csv",
        ]
    )
    payload["common"].update(
        {
            "calibration_split": "train",
            "selection_split": "validation",
            "test_split": "test",
            "selection_limit": 2,
            "rate_allocation": "global_exact",
            "include_global_single_component_controls": True,
            "two_stage_selection": True,
            "selection_top_k": 2,
            "skip_comfort": True,
        }
    )
    payload["stages"][0]["data_window_independence"] = (
        SUITE.TWO_STAGE_WINDOW_INDEPENDENCE
    )

    for overrides, message in (
        ({"selection_split": "train"}, "three distinct two-stage splits"),
        ({"selection_top_k": 1}, "selection_top_k >= 2"),
        ({"rate_allocation": "local_guard"}, "global_exact allocation"),
        (
            {"include_global_single_component_controls": False},
            "global no-joint controls",
        ),
    ):
        case = json.loads(json.dumps(payload))
        case["stages"][0]["runner_overrides"] = overrides
        config_path.write_text(json.dumps(case), encoding="utf-8")
        with pytest.raises(SUITE.SuiteConfigError, match=message):
            SUITE.load_suite_definition(config_path, repo_root=tmp_path)


def test_two_stage_allocator_uses_proxy_costs_for_nested_dominance() -> None:
    cap = 1024

    def report(
        strategy: str,
        *,
        source: str,
        final_cost: float,
        proxy_cost: float,
        natural: int,
    ) -> dict[str, object]:
        return {
            "mode": "global_exact",
            "endpoint_label": strategy,
            "selection_source": source,
            "strict_file_byte_feasible": True,
            "frontier_coarsening_events": 0,
            "fallback_policy": "forbidden_fail_closed",
            "q_l_cap_natural_file_bytes": cap,
            "selected_natural_file_bytes": natural,
            "unused_natural_bytes_before_tail_padding": cap - natural,
            "full_serializer_cross_checks": 3,
            "selected_hessian_cost": final_cost,
            "proxy_best_hessian_cost": proxy_cost,
        }

    controls = {
        "Q+S_OBS_global": report(
            "Q+S_OBS_global",
            source="global_exact_canonical_layout_pareto_frontier",
            final_cost=1.3,
            proxy_cost=1.1,
            natural=940,
        ),
        "Q+L_global": report(
            "Q+L_global",
            source="global_exact_canonical_layout_pareto_frontier",
            final_cost=1.2,
            proxy_cost=1.0,
            natural=960,
        ),
        "Q+S_OBS_or_L_global": report(
            "Q+S_OBS_or_L_global",
            source="global_exact_canonical_layout_exact_natural_dynamic_program",
            final_cost=0.9,
            proxy_cost=0.85,
            natural=980,
        ),
    }
    cap_best_nojoint = report(
        "Q+S_OBS_or_L_global",
        source="global_exact_canonical_layout_pareto_frontier",
        final_cost=0.8,
        proxy_cost=0.8,
        natural=950,
    )
    allocator = {
        "mode": "global_exact",
        "selection_source": "global_exact_canonical_layout_pareto_frontier",
        "strict_file_byte_feasible": True,
        "frontier_coarsening_events": 0,
        "q_l_cap_natural_file_bytes": cap,
        "selected_qsl_natural_file_bytes": 980,
        "unused_natural_bytes_before_tail_padding": 44,
        "full_serializer_cross_checks": 3,
        "selected_hessian_cost": 1.4,
        "proxy_best_hessian_cost": 0.7,
        "global_control_reports": controls,
        "nojoint_cap_best_audit": cap_best_nojoint,
        "joint_control_strategy": "Q+S_OBS_or_L_global",
        "joint_control_weakly_dominated": True,
        "joint_candidate_incremental_hessian_gain": 0.1,
        "nonjoint_union_weakly_dominates_pure_controls": True,
        "best_pure_control_hessian_cost": 1.0,
        "nonjoint_heterogeneous_gain_over_best_pure": 0.2,
        "joint_control_natural_match_available": True,
        "joint_control_required_natural_file_bytes": 980,
        "two_stage_selection": {
            "enabled": True,
            "proxy_top_k": 2,
            "rerank_metric": "validation_nll",
            "selection_split": "validation",
            "test_split_reserved_until_after_selection": True,
        },
    }
    arguments = {
        "rate_allocation": "global_exact",
        "include_global_single_component_controls": True,
        "two_stage_selection": True,
        "selection_top_k": 2,
        "selection_split": "validation",
    }
    SUITE._validate_global_allocator(
        {"rate_allocator": allocator, "selected_layers": ["a", "b"]},
        arguments,
    )

    bad = {**allocator, "proxy_best_hessian_cost": 0.9}
    with pytest.raises(SUITE.EvidenceError, match="nested no-joint"):
        SUITE._validate_global_allocator(
            {"rate_allocator": bad, "selected_layers": ["a", "b"]},
            arguments,
        )


def test_v4_config_uses_canonical_quantizers_and_matched_sentinel_grids() -> None:
    config_path = (
        REPO_ROOT / "configs" / "large_model_interaction_aware_v4_20260717.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["common"]["candidate_quantizers"] == [
        "symmetric_mse_clip",
        "symmetric_rtn",
    ]
    expected_sentinel_overrides = {
        "calib_limit": 4,
        "selection_limit": 2,
        "eval_limit": 2,
        "candidate_lowrank_factor_bits": [4, 16],
        "candidate_family_top_k": 1,
        "selection_top_k": 2,
        "global_frontier_top_ranks": 1,
        "global_frontier_support_fractions": [0.75],
        "global_frontier_budget_multipliers": [1.25],
        "repair_block_sizes": [512],
        "max_allocation_ranks": 32,
        "allocation_rank_grid": [0, 1, 2, 4, 8, 16, 32],
    }
    assert payload["resource_policy"]["sentinel_timeout_hours"] == 16
    sentinel_ids = {
        "s0_qwen_two_stage_sentinel",
        "s0_llama_two_stage_sentinel",
        "s0_mistral_two_stage_sentinel",
    }
    for stage_id in (
        "s0_qwen_two_stage_sentinel",
        "s0_llama_two_stage_sentinel",
        "s0_mistral_two_stage_sentinel",
    ):
        sentinel = next(
            stage for stage in payload["stages"] if stage["id"] == stage_id
        )
        assert sentinel["runner_overrides"] == expected_sentinel_overrides

    definition = SUITE.load_suite_definition(config_path, repo_root=REPO_ROOT)
    sentinel_jobs = [
        job
        for job in SUITE.expand_jobs(definition, environment={})
        if job.stage_id in sentinel_ids
    ]
    assert {job.stage_id for job in sentinel_jobs} == sentinel_ids
    assert len(sentinel_jobs) == 3
    for job in sentinel_jobs:
        for key, value in expected_sentinel_overrides.items():
            assert job.effective_arguments[key] == value
        command = SUITE.build_runner_command(
            job,
            python_executable="python",
            runner=definition.runner,
        )

        def command_value(flag: str) -> str:
            assert command.count(flag) == 1
            return command[command.index(flag) + 1]

        expected_cli_values = {
            "--calib-limit": "4",
            "--selection-limit": "2",
            "--eval-limit": "2",
            "--candidate-lowrank-factor-bits": "4,16",
            "--candidate-family-top-k": "1",
            "--selection-top-k": "2",
            "--global-frontier-top-ranks": "1",
            "--global-frontier-support-fractions": "0.75",
            "--global-frontier-budget-multipliers": "1.25",
            "--repair-block-sizes": "512",
            "--max-allocation-ranks": "32",
            "--allocation-rank-grid": "0,1,2,4,8,16,32",
        }
        for flag, expected in expected_cli_values.items():
            assert command_value(flag) == expected
        assert command.count("--two-stage-selection") == 1
        assert command.count("--include-global-single-component-controls") == 1


def test_two_stage_evidence_keeps_validation_proxies_out_of_final_test(
    tmp_path: Path,
) -> None:
    output = tmp_path / "job"
    output.mkdir()
    strategies = [
        "Q+S_OBS_global",
        "Q+L_global",
        "Q+S_OBS_or_L_global",
        "Q+S+L_QL_budget",
    ]
    selection_reports = {}
    rerank_rows = [
        {
            "strategy": "dense_validation",
            "proxy_rank": 0,
            "allocation_digest": "dense",
            "hessian_cost": 0,
            "natural_file_bytes": 0,
            "validation_nll": 1.0,
            "validation_perplexity": 2.0,
            "validation_tokens": 30,
            "validation_nll_delta": 0,
            "selected_by_validation": False,
        }
    ]
    for strategy in strategies:
        digest = hashlib.sha256(strategy.encode()).hexdigest()
        selection_reports[strategy] = {
            "validation_selected_proxy_rank": 1,
            "validation_selected_allocation_digest": digest,
        }
        rerank_rows.append(
            {
                "strategy": strategy,
                "proxy_rank": 1,
                "allocation_digest": digest,
                "hessian_cost": 1.0,
                "natural_file_bytes": 100,
                "validation_nll": 1.1,
                "validation_perplexity": 3.0,
                "validation_tokens": 30,
                "validation_nll_delta": 0.1,
                "selected_by_validation": True,
            }
        )
    with (output / "allocation_validation_rerank.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rerank_rows[0]))
        writer.writeheader()
        writer.writerows(rerank_rows)
    validation_windows = [
        {
            "strategy": "dense_validation",
            "base_strategy": "",
            "evidence_role": "allocation_validation",
        }
    ]
    validation_windows.extend(
        {
            "strategy": f"{strategy}__proxy_rank_1",
            "base_strategy": strategy,
            "evidence_role": "allocation_validation",
        }
        for strategy in strategies
    )
    with (output / "allocation_validation_window_nll.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_windows[0]))
        writer.writeheader()
        writer.writerows(validation_windows)
    endpoint_windows = [
        {"strategy": strategy, "evidence_role": "final_test"}
        for strategy in ["dense", *strategies]
    ]
    with (output / "endpoint_window_nll.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_windows[0]))
        writer.writeheader()
        writer.writerows(endpoint_windows)

    assignment = json.dumps({"layer.0": 4})
    endpoint_rows = {
        strategy: {
            "validation_selected_proxy_rank": "1",
            "q_bits_by_layer": assignment,
            "q_quantizers_by_layer": json.dumps(
                {"layer.0": "symmetric_rtn"}
            ),
            "q_group_sizes_by_layer": json.dumps({"layer.0": 0}),
            "lowrank_factor_bits_by_layer": json.dumps({"layer.0": 16}),
        }
        for strategy in strategies
    }
    job = SimpleNamespace(
        output_dir=output,
        expected_strategies=tuple(strategies),
        tensor_scope={"expected_selected_tensors": 1},
        effective_arguments={
            "two_stage_selection": True,
            "calibration_split": "train",
            "selection_split": "validation",
            "test_split": "test",
            "selection_limit": 2,
            "selection_top_k": 2,
            "sequence_length": 16,
            "include_global_single_component_controls": True,
        },
    )
    data = {
        "role_splits": {
            "calibration": "train",
            "selection": "validation",
            "test": "test",
        },
        "test_reserved_until_after_validation_selection": True,
        "calibration_selection_identical_text_overlap_count": 0,
        "identical_text_overlap_count": 0,
        "selection_test_identical_text_overlap_count": 0,
        "calib_digest": "a" * 64,
        "selection_digest": "b" * 64,
        "eval_digest": "c" * 64,
    }
    run_config = {
        "allocation_validation_baseline_metrics": {
            "nll": 1.0,
            "perplexity": 2.0,
            "tokens": 30,
        },
        "rate_allocator": {
            "two_stage_selection": {
                "selection_reports": selection_reports,
            }
        },
    }
    evidence = SUITE._validate_two_stage_selection_evidence(
        job, run_config, data, endpoint_rows
    )
    assert evidence["test_reserved_until_after_validation_selection"] is True

    endpoint_windows[-1]["strategy"] = (
        "Q+S+L_QL_budget__proxy_rank_1"
    )
    with (output / "endpoint_window_nll.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_windows[0]))
        writer.writeheader()
        writer.writerows(endpoint_windows)
    with pytest.raises(SUITE.EvidenceError, match="proxy labels leaked"):
        SUITE._validate_two_stage_selection_evidence(
            job, run_config, data, endpoint_rows
        )


def test_declared_endpoint_identity_requires_hash_natural_bytes_and_metrics() -> None:
    job = SimpleNamespace(
        tensor_scope={"endpoint_identity_pairs": [["Q+L", "Q+L_global"]]}
    )
    metric_fields = (
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
    rows = {
        strategy: {field: "1.0" for field in metric_fields}
        for strategy in ("Q+L", "Q+L_global")
    }
    artifacts = {
        strategy: {
            "sha256": "a" * 64,
            "file_bytes": 1024,
            "natural_file_bytes": 960,
        }
        for strategy in ("Q+L", "Q+L_global")
    }
    evidence = SUITE._validate_endpoint_identities(job, rows, artifacts)
    assert evidence[0]["endpoint_metrics_identical"] is True

    bad_hash = {key: dict(value) for key, value in artifacts.items()}
    bad_hash["Q+L_global"]["sha256"] = "b" * 64
    with pytest.raises(SUITE.EvidenceError, match="artifact sha256"):
        SUITE._validate_endpoint_identities(job, rows, bad_hash)

    bad_rows = {key: dict(value) for key, value in rows.items()}
    bad_rows["Q+L_global"]["heldout_nll"] = "1.1"
    with pytest.raises(SUITE.EvidenceError, match="heldout_nll"):
        SUITE._validate_endpoint_identities(job, bad_rows, artifacts)


def test_resource_evidence_binds_physical_gpu_gate_runtime_and_runner(
    tmp_path: Path,
) -> None:
    policy = {
        "enforce_at_runtime": True,
        "eligible_physical_gpus": [0, 1],
        "preferred_physical_gpu": 1,
        "gpu_memory_threshold_mib": 1024,
        "gpu_utilization_threshold_percent": 10,
        "sample_interval_seconds": 30,
        "gate_wait_timeout_hours": 24,
        "minimum_available_host_memory_gib": 256,
        "minimum_output_disk_free_gib": 100,
        "sentinel_timeout_hours": 4,
        "three_tensor_pair_timeout_hours": 16,
        "maximum_gpu_memory_mib": 77824,
        "maximum_rss_gib": 384,
    }
    gate_path = tmp_path / "job.gate.json"
    runtime_path = tmp_path / "job.runtime.json"
    job = SimpleNamespace(
        resource_policy=policy,
        resource_gate_path=gate_path,
        resource_runtime_path=runtime_path,
        suite_id="suite",
        job_id="job",
        suite_config_sha256="1" * 64,
        job_config_sha256="2" * 64,
        execution_fingerprint_sha256="3" * 64,
        tensor_scope={"expected_selected_tensors": 1},
    )
    gpu_sample = {
        "sampled_at": "2026-07-16T00:00:00Z",
        "monotonic_seconds": 100.0,
        "physical_gpu": 1,
        "memory_used_mib": 17,
        "utilization_gpu_percent": 0,
    }
    gpu_sample_later = {**gpu_sample, "monotonic_seconds": 130.0}
    host_sample = {
        "sampled_at": "2026-07-16T00:00:00Z",
        "available_host_memory_gib": 512.0,
        "output_disk_probe_path": str(tmp_path),
        "output_disk_free_gib": 500.0,
    }
    gate = {
        "schema_version": SUITE.RESOURCE_GATE_SCHEMA_VERSION,
        "suite_id": job.suite_id,
        "job_id": job.job_id,
        "suite_config_sha256": job.suite_config_sha256,
        "job_config_sha256": job.job_config_sha256,
        "execution_fingerprint_sha256": job.execution_fingerprint_sha256,
        "policy_sha256": SUITE._object_sha256(policy),
        "policy": policy,
        "selected_physical_gpu": 1,
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": "1",
        "lock_path": "/tmp/com_compression_gpu_1.lock",
        "lock_acquired": True,
        "gate_passed": True,
        "pre_lock_samples": [gpu_sample, gpu_sample_later],
        "post_lock_samples": [gpu_sample, gpu_sample_later],
        "host_pre_lock": host_sample,
        "host_post_lock": host_sample,
    }
    SUITE._write_json_atomic(gate_path, gate)
    gate_sha = _sha(gate_path)
    runtime = {
        "schema_version": SUITE.RESOURCE_RUNTIME_SCHEMA_VERSION,
        "suite_id": job.suite_id,
        "job_id": job.job_id,
        "suite_config_sha256": job.suite_config_sha256,
        "job_config_sha256": job.job_config_sha256,
        "execution_fingerprint_sha256": job.execution_fingerprint_sha256,
        "selected_physical_gpu": 1,
        "resource_gate_sha256": gate_sha,
        "timed_out": False,
        "runner_exit_code": 0,
        "monitor_errors": [],
        "limits_passed": True,
        "timeout_seconds": 4 * 3600.0,
        "sample_interval_seconds": 5.0,
        "sample_count": 1,
        "gpu_samples": [gpu_sample],
        "peak_gpu_memory_mib": 17,
        "peak_gpu_utilization_percent": 0,
        "child_max_rss_gib": 32.0,
        "child_rss_measurement": "RUSAGE_CHILDREN.ru_maxrss upper bound",
        "maximum_gpu_memory_mib": 77824,
        "maximum_rss_gib": 384.0,
    }
    SUITE._write_json_atomic(runtime_path, runtime)
    run_config = {
        "arguments": {"resource_gate_manifest": str(gate_path)},
        "resource_gate": {
            "schema_version": SUITE.RESOURCE_GATE_SCHEMA_VERSION,
            "path": str(gate_path.resolve()),
            "sha256": gate_sha,
            "selected_physical_gpu": 1,
            "consumed_before_model_load": True,
        },
        "runtime": {"cuda_visible_devices": "1"},
    }
    evidence = SUITE._validate_resource_evidence(job, run_config)
    assert evidence["selected_physical_gpu"] == 1
    assert evidence["limits_passed"] is True

    runtime["timeout_seconds"] = 0.0
    SUITE._write_json_atomic(runtime_path, runtime)
    with pytest.raises(SUITE.EvidenceError, match="timeout differs"):
        SUITE._validate_resource_evidence(job, run_config)
    runtime["timeout_seconds"] = 4 * 3600.0

    runtime["sample_interval_seconds"] = 30.0
    SUITE._write_json_atomic(runtime_path, runtime)
    with pytest.raises(SUITE.EvidenceError, match="monitor interval differs"):
        SUITE._validate_resource_evidence(job, run_config)
    runtime["sample_interval_seconds"] = 5.0

    runtime["peak_gpu_memory_mib"] = 0
    SUITE._write_json_atomic(runtime_path, runtime)
    with pytest.raises(SUITE.EvidenceError, match="peak GPU memory differs"):
        SUITE._validate_resource_evidence(job, run_config)
    runtime["peak_gpu_memory_mib"] = 17
    SUITE._write_json_atomic(runtime_path, runtime)

    gate["post_lock_samples"][1] = {
        **gpu_sample,
        "monotonic_seconds": 129.0,
    }
    SUITE._write_json_atomic(gate_path, gate)
    with pytest.raises(SUITE.EvidenceError, match="configured sample interval"):
        SUITE._validate_resource_evidence(job, run_config)
    gate["post_lock_samples"][1] = gpu_sample_later
    SUITE._write_json_atomic(gate_path, gate)

    gate["post_lock_samples"][1] = {
        **gpu_sample_later,
        "memory_used_mib": -1,
    }
    SUITE._write_json_atomic(gate_path, gate)
    with pytest.raises(SUITE.EvidenceError, match="invalid GPU memory"):
        SUITE._validate_resource_evidence(job, run_config)
    gate["post_lock_samples"][1] = gpu_sample_later
    SUITE._write_json_atomic(gate_path, gate)

    gate["post_lock_samples"][1] = {
        **gpu_sample_later,
        "utilization_gpu_percent": 10,
    }
    SUITE._write_json_atomic(gate_path, gate)
    with pytest.raises(SUITE.EvidenceError, match="exceeds the launch threshold"):
        SUITE._validate_resource_evidence(job, run_config)


def test_command_is_explicit_about_physical_rate_scope_and_does_not_write(tmp_path: Path) -> None:
    definition = SUITE.load_suite_definition(
        REPO_ROOT / "configs" / "large_scale_hessian_suite_20260714.json",
        repo_root=REPO_ROOT,
        output_root_override=tmp_path / "never_created",
    )
    job = next(
        item
        for item in SUITE.expand_jobs(definition, environment={})
        if item.job_id == "pythia70m_full_mlp_scalability__seed17__rate0p258"
    )
    command = SUITE.build_runner_command(job, python_executable="python", runner=definition.runner)
    joined = " ".join(command)
    assert "--layers 0,1,2,3,4,5" in joined
    assert "--max-modules 0" in joined
    assert "--target-ratios 0.258 --endpoint-target 0.258" in joined
    assert "--emit-codec-artifacts" in command
    assert "--enforce-serialized-rate-cap" in command
    dry = SUITE._dry_run_payload(
        [job], [], python_executable="python", runner=definition.runner, resume=False
    )
    assert dry["writes_performed"] is False
    assert dry["selected_jobs"][0]["action"] == "would_execute"
    assert not (tmp_path / "never_created").exists()


def test_successful_lifecycle_is_resumable_only_after_full_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    definition, job = _make_fake_suite(tmp_path)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _write_fake_completed_output(job)
        stdout = kwargs["stdout"]
        stdout.write("fake success\n")
        stdout.flush()
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(SUITE, "_run_process", fake_run)
    evidence = SUITE.execute_job(
        definition,
        job,
        python_executable=sys.executable,
        runner=definition.runner,
    )
    assert evidence["actual_eval_tokens"] == 45
    assert evidence["selected_tensor_count"] == 2
    inspection = SUITE.inspect_job(job)
    assert inspection.status == "completed_valid"
    assert inspection.state["exit_code"] == 0
    assert inspection.state["suite_config_sha256"] == definition.config_sha256
    assert inspection.state["execution_fingerprint_sha256"] == job.execution_fingerprint_sha256
    assert (job.output_dir / "_suite_job_record.json").is_file()

    artifact = job.output_dir / "artifacts" / "strategy_0.bin"
    artifact.write_bytes(b"tampered")
    invalid = SUITE.inspect_job(job)
    assert invalid.status == "invalid"
    assert "mismatch" in invalid.reason


def test_protocol_lifecycle_enables_consumption_and_aggregation_only_after_output_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    definition = SUITE.load_suite_definition(_minimal_protocol_config(tmp_path), repo_root=tmp_path)
    job = SUITE.expand_jobs(definition, environment={})[0]

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        running = json.loads(job.state_path.read_text(encoding="utf-8"))
        assert running["protocol_manifest_consumed"] is False
        assert running["seed_aggregation_allowed"] is False
        _write_fake_completed_output(job)
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(SUITE, "_run_process", fake_run)
    evidence = SUITE.execute_job(
        definition,
        job,
        python_executable=sys.executable,
        runner=definition.runner,
    )
    record = json.loads((job.output_dir / "_suite_job_record.json").read_text(encoding="utf-8"))
    assert record["protocol_manifest_consumed"] is True
    assert record["seed_aggregation_allowed"] is True
    assert SUITE.inspect_job(job).status == "completed_valid"
    assert set(evidence["expected_output_files"]) == {
        "COMPLETED",
        *job.expected_outputs,
    }

    endpoint_path = job.output_dir / "endpoint_window_nll.csv"
    original_endpoint = endpoint_path.read_bytes()
    with endpoint_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["nll"] = "9.0"
    with endpoint_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    assert SUITE.inspect_job(job).status == "invalid"
    endpoint_path.write_bytes(original_endpoint)
    assert SUITE.inspect_job(job).status == "completed_valid"

    run_path = job.output_dir / "run_config.json"
    original_run = run_path.read_bytes()
    run_config = json.loads(original_run)
    run_config["baseline_metrics"]["nll"] = 9.0
    run_path.write_text(json.dumps(run_config), encoding="utf-8")
    assert SUITE.inspect_job(job).status == "invalid"
    run_path.write_bytes(original_run)
    assert SUITE.inspect_job(job).status == "completed_valid"


def test_running_and_failed_states_are_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    definition, job = _make_fake_suite(tmp_path)

    def fake_failure(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 7)

    monkeypatch.setattr(SUITE, "_run_process", fake_failure)
    with pytest.raises(SUITE.EvidenceError, match="exited with code 7"):
        SUITE.execute_job(
            definition,
            job,
            python_executable=sys.executable,
            runner=definition.runner,
        )
    failed = SUITE.inspect_job(job)
    assert failed.status == "failed"
    assert (job.output_dir / "FAILED").is_file()

    state = json.loads(job.state_path.read_text(encoding="utf-8"))
    state["status"] = "RUNNING"
    state["exit_code"] = None
    job.state_path.write_text(json.dumps(state), encoding="utf-8")
    running = SUITE.inspect_job(job)
    assert running.status == "running"
    assert "fail-closed" in running.reason


def test_suite_outputs_separate_planned_from_verified_results(tmp_path: Path) -> None:
    definition, job = _make_fake_suite(tmp_path)
    inspections = {job.job_id: SUITE.inspect_job(job)}
    SUITE.write_suite_outputs(
        definition,
        [job],
        inspections,
        python_executable=sys.executable,
        runner=definition.runner,
    )
    manifest = json.loads((definition.output_root / "suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status_counts"] == {
        "completed_valid": 0,
        "failed": 0,
        "invalid": 0,
        "planned": 1,
        "running": 0,
    }
    assert manifest["jobs"][0]["actual"] is None
    assert manifest["jobs"][0]["planned_eval_nll_tokens"] == 45
    assert "not a claim that planned jobs ran" in (
        definition.output_root / "suite_summary.md"
    ).read_text(encoding="utf-8")
    report = SUITE.check_suite(definition, [job], [job])
    assert report["ok"] is True
