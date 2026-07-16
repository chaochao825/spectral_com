from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_liftquant_official",
    REPO_ROOT / "scripts" / "audit_liftquant_official.py",
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _init_git(repo: Path) -> str:
    _git(repo, "init", "--quiet")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Audit Fixture",
        "-c",
        "user.email=audit-fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    return _git(repo, "rev-parse", "HEAD")


def test_fake_checkout_detects_cli_path_and_e2e_failures(tmp_path: Path) -> None:
    repo = tmp_path / "LiftQuant"
    repo.mkdir()
    _write(
        repo / "README.md",
        """# fixture

```bash
python main.py \\
  --model fixture \\
  --nsamples 4096 \\
  --epochs 2
```

```bash
python e2efinetune.py --fp_model_path fixture --num_train_epochs 1
```
""",
    )
    _write(
        repo / "main.py",
        """import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--nsamples1', type=int, default=128)
parser.add_argument('--nsamples2', type=int, default=128)
parser.add_argument('--epochs1', type=int, default=10)
parser.add_argument('--epochs2', type=int, default=10)
parser.parse_args()
""",
    )
    _write(
        repo / "e2efinetune.py",
        """from dataclasses import dataclass
from datautils_block import test_ppl

@dataclass
class ModelArguments:
    fp_model_path: str = ''

def train():
    model_args, extra_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
""",
    )
    missing_path = "/mnt/liftquant-audit-fixture/redpajama_cache"
    _write(
        repo / "datautils.py",
        f"REDPAJAMA_CACHE = '{missing_path}'\n",
    )
    _write(repo / "requirements.txt", "pytest\n")

    audit = AUDIT.build_audit(
        repo,
        python=Path(sys.executable),
        expected_commit=None,
        timeout_seconds=10,
    )

    assert audit["entrypoints"]["main"]["help_probe"]["returncode"] == 0
    assert audit["entrypoints"]["main"]["unsupported_readme_flags"] == [
        "--epochs",
        "--nsamples",
    ]
    assert audit["entrypoints"]["e2e"]["help_probe"]["status"] == "failed"
    assert audit["entrypoints"]["e2e"]["missing_local_imports"] == [
        {"file": "e2efinetune.py", "line": 2, "module": "datautils_block"}
    ]
    assert audit["paths"]["hardcoded_redpajama"] == [
        {
            "file": "datautils.py",
            "line": 1,
            "path": missing_path,
            "exists_on_audit_host": False,
        }
    ]
    assert not audit["execution"]["training_executed"]
    assert not audit["execution"]["gpu_job_executed"]
    assert audit["provenance"]["provenance_verified"] is False
    assert audit["provenance"]["worktree_clean"] is False
    assert audit["provenance"]["source_worktree_clean"] is False
    assert "GIT_PROVENANCE_PROBE_FAILED" in {
        finding["id"] for finding in audit["findings"]
    }

    paths = AUDIT.write_outputs(audit, tmp_path / "out")
    assert {path.name for path in paths} == {"audit.json", "summary.md"}
    reloaded = json.loads((tmp_path / "out" / "audit.json").read_text(encoding="utf-8"))
    assert reloaded["compatibility"]["external_reproduction"] == "pending"
    assert "没有启动训练或量化" in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")


def test_patched_smoke_completion_gate_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write(run_dir / "exit_code", "0\n")
    _write(run_dir / "status", "COMPLETED\n")
    _write(run_dir / "time.txt", "elapsed=1.25 maxrss_kb=42\n")
    _write(
        run_dir / "stdout.log",
        "=== Start quantize layer 0 ===\n"
        "saved 20to8-layer0.pth\n"
        "(main.py 123): INFO 1.25\n",
    )
    _write(run_dir / "stderr.log", "")

    missing_artifact = AUDIT.inspect_smoke_run(
        run_dir, source_variant="patched", physical_gpu="2"
    )
    gate = AUDIT.smoke_run_completion_gate(missing_artifact)
    assert gate["passed"] is False
    assert gate["checks"]["nonempty_hashed_layer0_artifact"] is False

    _write(run_dir / "qmodels" / "Qwen2.5-3B+20to8-layer0.pth", "artifact\n")
    complete = AUDIT.inspect_smoke_run(
        run_dir, source_variant="patched", physical_gpu="2"
    )
    gate = AUDIT.smoke_run_completion_gate(complete)
    assert gate["passed"] is True
    assert all(gate["checks"].values())


def test_git_provenance_does_not_hide_source_diff_after_truncated_cache_noise(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "main.py", "print('clean')\n")
    head = _init_git(repo)
    for index in range(100):
        _write(
            repo / "very_long_package_name" / "__pycache__" / f"cache_{index:03d}_with_a_long_name.cpython-313.pyc",
            "cache\n",
        )
    _write(repo / "main.py", "print('source changed')\n")

    info = AUDIT._git_info(repo, head, timeout_seconds=10)

    assert info["provenance_verified"] is True
    assert info["commit_matches"] is True
    assert info["worktree_clean"] is False
    assert info["status_probe"]["stdout_truncated"] is True
    assert info["source_worktree_clean"] is False
    assert any("main.py" in entry for entry in info["source_dirty_entries"])


def test_block_lane_fails_closed_for_wrong_commit_and_failed_help(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "README.md", "```bash\npython main.py --model fixture\n```\n")
    _write(repo / "main.py", "raise RuntimeError('help import failed')\n")
    _write(repo / "requirements.txt", "")
    _init_git(repo)

    audit = AUDIT.build_audit(
        repo,
        python=Path(sys.executable),
        expected_commit="0" * 40,
        timeout_seconds=10,
    )

    assert audit["provenance"]["provenance_verified"] is True
    assert audit["provenance"]["commit_matches"] is False
    assert audit["entrypoints"]["main"]["help_probe"]["status"] == "failed"
    assert audit["compatibility"]["block_correction"] == "blocked_pending_upstream_or_local_adapter"
    assert "MAIN_HELP_PROBE_FAILED" in {finding["id"] for finding in audit["findings"]}


def test_committed_official_audit_has_complete_block_and_e2e_evidence() -> None:
    result_dir = REPO_ROOT / "results" / "liftquant_official_integration_20260714"
    audit = json.loads((result_dir / "audit.json").read_text(encoding="utf-8"))
    summary = (result_dir / "summary.md").read_text(encoding="utf-8")

    assert audit["provenance"]["commit"] == AUDIT.EXPECTED_COMMIT
    assert audit["provenance"]["commit_matches"] is True
    assert audit["provenance"]["provenance_verified"] is True
    assert audit["provenance"]["source_worktree_clean"] is True
    for probe_name in ("head_probe", "status_probe", "source_status_probe"):
        probe = audit["provenance"][probe_name]
        assert probe["returncode"] == 0
        assert probe["timed_out"] is False
    assert audit["entrypoints"]["main"]["help_probe"]["returncode"] == 0
    assert audit["entrypoints"]["main"]["unsupported_readme_flags"] == [
        "--epochs",
        "--nsamples",
    ]
    runtime_flags = audit["entrypoints"]["main"]["help_probe"]["runtime_flags"]
    assert "--epochs" not in runtime_flags
    assert "--nsamples" not in runtime_flags
    e2e = audit["entrypoints"]["e2e"]
    assert e2e["help_probe"]["returncode"] != 0
    assert any(row["module"] == "datautils_block" for row in e2e["missing_local_imports"])

    hardcoded = audit["paths"]["hardcoded_redpajama"]
    assert {row["path"] for row in hardcoded} >= {
        "/mnt/bn/adsinfra-gpu-dev-hl/heliulu/datasets/redpajama_cache",
        "/data/shared_data/datasets",
    }
    assert all(row["exists_on_audit_host"] is False for row in hardcoded)

    requirements = {row["name"]: row for row in audit["requirements"]["comparison"]}
    assert requirements["transformers"]["exact_pin"] == "5.9.0"
    assert requirements["transformers"]["status"] == "version_mismatch"
    assert requirements["bitblas"]["status"] == "missing"
    assert {"accelerate", "datasets", "numpy", "scipy"} <= set(
        audit["requirements"]["imported_but_not_declared"]
    )

    assert audit["model_cache"]["minimal_files_present"] is True
    assert audit["model_cache"]["missing_referenced_shards"] == []
    assert all(
        row["safetensors_header_json_valid"]
        for row in audit["model_cache"]["shard_checks"]
    )
    assert audit["sample_accounting"]["stage1_reduces_equal_max_by_one_thirty_second"] is True
    assert audit["sample_accounting"]["stage2_reduces_equal_max_by_one_thirty_second"] is True
    assert audit["compatibility"]["external_reproduction"] == "pending"
    assert audit["compatibility"]["accuracy_claim_allowed"] is False
    assert audit["compatibility"]["patched_layer0_smoke"] == "control_flow_and_artifact_passed"
    smoke = audit["patched_smoke_evidence"]
    patch = smoke["compatibility_patch"]
    assert patch["sha256"] == "47d5437744873f9d2b65074ebf4a07322f4a92d73ea6bf427ce5b2afc6f7d7a2"
    assert patch["git_apply_check"]["returncode"] == 0
    assert patch["mode_metadata_matches_clean_target"] is True
    assert smoke["official_unpatched_run"]["exit_code"] == 1
    assert "attention_type" in smoke["official_unpatched_run"]["error_signature"]
    patched = smoke["patched_run"]
    assert smoke["patched_run_completion_gate"]["passed"] is True
    assert all(smoke["patched_run_completion_gate"]["checks"].values())
    assert patched["exit_code"] == 0
    assert patched["status_file"] == "COMPLETED"
    assert patched["elapsed_seconds"] == 195.71
    assert patched["maxrss_kb"] == 9_000_700
    assert patched["stdout_sha256"] == "ea1f8a3f8778d283ae2fb7181d2b06c80472e072b09d4eb21ac4ae651a3ab1d5"
    assert patched["stderr_sha256"] == "59daa2d797699a16574e0e8236300dfc36336c706550a5f0cc5cb9852f75a793"
    assert patched["time_sha256"] == "2ec2d343d69ad7689fd0fa725d1417c21eafab29a3732ead43bad8b5a72f2774"
    assert patched["scope"] == {
        "model": "Qwen2.5-3B",
        "mapping": "20to8",
        "wbits": "2",
        "seqlen": "128",
        "nsamples1": "8",
        "nsamples2": "8",
        "epochs1": "1",
        "epochs2": "1",
        "batch_size": "2",
        "quant_start": "0",
        "quant_end": "1",
        "eval_ppl": False,
        "tasks": "",
    }
    layer0 = next(
        row for row in patched["artifacts"] if row["relative_path"].endswith("20to8-layer0.pth")
    )
    assert layer0["size_bytes"] == 24_566_073
    assert layer0["sha256"] == "5382f0f65da351df04ae2c84b028b2c3ad18370b720f68d1fc007f27ddefa5e1"
    assert audit["commands"]["reduced_qwen_smoke"]["status"] == "template_only_not_executed"
    assert audit["commands"]["block_correction_full_after_path_fix"]["status"] == "blocked_template_not_executed"
    assert audit["execution"] == {
        "scope": "audit_process_only",
        "does_not_describe_external_smoke_evidence": True,
        "training_executed": False,
        "quantization_executed": False,
        "gpu_job_executed": False,
        "model_loaded": False,
        "dataset_loaded": False,
        "subprocess_policy": "git metadata, importlib.metadata, and --help only; shell=False; bounded timeout",
    }
    finding_ids = {finding["id"] for finding in audit["findings"]}
    assert {
        "README_MAIN_FLAGS_NOT_IMPLEMENTED",
        "REDPAJAMA_CACHE_NOT_PORTABLE",
        "E2E_ENTRYPOINT_IMPORT_FAILURE",
        "OFFICIAL_QWEN_SMOKE_COMPATIBILITY_FAILURE",
        "PATCHED_LAYER0_SMOKE_IS_NOT_FULL_REPRODUCTION",
    } <= finding_ids
    assert summary == AUDIT.render_summary(audit)
    assert "external_reproduction_pending" in summary
    assert "Block correction" in summary
    assert "Optional E2E" in summary
