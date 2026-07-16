#!/usr/bin/env python3
"""Audit a pinned LiftQuant checkout without launching model training.

The audit is intentionally limited to static AST/text inspection and short,
CPU-only subprocesses (git metadata, installed-package metadata, and ``--help``).
It never imports LiftQuant into this process and never launches a CUDA job.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


VERIFIED_AS_OF = "2026-07-14"
EXPECTED_COMMIT = "72b3875c770e4579639931fed89dc95e4067edac"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "liftquant_official_integration_20260714"
)
MAX_CAPTURE_CHARS = 4_000


def _run_limited(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one bounded command without a shell and return sanitized evidence."""

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_env:
        env.update(extra_env)
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return {
            "command": [str(part) for part in command],
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout_excerpt": stdout[:MAX_CAPTURE_CHARS],
            "stderr_excerpt": stderr[:MAX_CAPTURE_CHARS],
            "stdout_nonempty": bool(stdout),
            "stderr_nonempty": bool(stderr),
            "stdout_length_chars": len(stdout),
            "stderr_length_chars": len(stderr),
            "stdout_line_count": len(stdout.splitlines()),
            "stderr_line_count": len(stderr.splitlines()),
            "stdout_truncated": len(stdout) > MAX_CAPTURE_CHARS,
            "stderr_truncated": len(stderr) > MAX_CAPTURE_CHARS,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "command": [str(part) for part in command],
            "returncode": None,
            "timed_out": True,
            "stdout_excerpt": stdout[:MAX_CAPTURE_CHARS],
            "stderr_excerpt": stderr[:MAX_CAPTURE_CHARS],
            "stdout_nonempty": bool(stdout),
            "stderr_nonempty": bool(stderr),
            "stdout_length_chars": len(stdout),
            "stderr_length_chars": len(stderr),
            "stdout_line_count": len(stdout.splitlines()),
            "stderr_line_count": len(stderr.splitlines()),
            "stdout_truncated": len(stdout) > MAX_CAPTURE_CHARS,
            "stderr_truncated": len(stderr) > MAX_CAPTURE_CHARS,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        }
    except OSError as exc:
        return {
            "command": [str(part) for part in command],
            "returncode": None,
            "timed_out": False,
            "stdout_excerpt": "",
            "stderr_excerpt": f"{type(exc).__name__}: {exc}",
            "stdout_nonempty": False,
            "stderr_nonempty": True,
            "stdout_length_chars": 0,
            "stderr_length_chars": len(str(exc)),
            "stdout_line_count": 0,
            "stderr_line_count": 1,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _literal(value: ast.AST | None) -> Any:
    if value is None:
        return None
    try:
        result = ast.literal_eval(value)
    except (ValueError, TypeError):
        return ast.unparse(value) if hasattr(ast, "unparse") else type(value).__name__
    if isinstance(result, (str, int, float, bool, type(None))):
        return result
    if isinstance(result, (list, tuple)):
        return list(result)
    return repr(result)


def argparse_flags_from_ast(path: Path) -> list[dict[str, Any]]:
    """Extract literal ``ArgumentParser.add_argument`` options without imports."""

    if not path.exists():
        return []
    tree = ast.parse(_read_text(path), filename=str(path))
    records: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
            continue
        flags = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        ]
        if not flags:
            continue
        keywords = {keyword.arg: _literal(keyword.value) for keyword in node.keywords if keyword.arg}
        records.append(
            {
                "flags": flags,
                "line": node.lineno,
                "default": keywords.get("default"),
                "action": keywords.get("action"),
                "choices": keywords.get("choices"),
            }
        )
    return sorted(records, key=lambda item: (item["line"], item["flags"]))


def dataclass_flags_from_ast(path: Path) -> list[dict[str, Any]]:
    """Extract fields declared in local dataclasses used by HfArgumentParser."""

    if not path.exists():
        return []
    tree = ast.parse(_read_text(path), filename=str(path))
    records: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = {
            decorator.id
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Name)
        }
        if "dataclass" not in decorators:
            continue
        bases = [ast.unparse(base) if hasattr(ast, "unparse") else type(base).__name__ for base in node.bases]
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                continue
            default: Any = None
            if isinstance(statement.value, ast.Call):
                default_keyword = next(
                    (keyword.value for keyword in statement.value.keywords if keyword.arg == "default"),
                    None,
                )
                default = _literal(default_keyword)
            else:
                default = _literal(statement.value)
            records.append(
                {
                    "flag": "--" + statement.target.id,
                    "field": statement.target.id,
                    "class": node.name,
                    "class_bases": bases,
                    "line": statement.lineno,
                    "default": default,
                }
            )
    return records


def extract_readme_commands(readme: str) -> list[dict[str, Any]]:
    """Return fenced Python command blocks and their literal long options."""

    commands: list[dict[str, Any]] = []
    for index, match in enumerate(re.finditer(r"```[^\n]*\n(.*?)```", readme, flags=re.DOTALL), start=1):
        block = match.group(1).strip()
        entry_match = re.search(r"\bpython\s+([^\s\\]+\.py)\b", block)
        if not entry_match:
            continue
        commands.append(
            {
                "fence_index": index,
                "entrypoint": entry_match.group(1),
                "flags": sorted(set(re.findall(r"--[A-Za-z0-9_-]+", block))),
                "command": block,
            }
        )
    return commands


def parse_requirements(path: Path) -> list[dict[str, str | None]]:
    if not path.exists():
        return []
    rows: list[dict[str, str | None]] = []
    pattern = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(?:==\s*([^\s;]+))?")
    for line_number, raw in enumerate(_read_text(path).splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-r", "--")):
            continue
        match = pattern.match(line)
        if match:
            rows.append(
                {
                    "name": match.group(1),
                    "normalized_name": normalize_distribution(match.group(1)),
                    "exact_pin": match.group(2),
                    "raw": line,
                    "line": str(line_number),
                }
            )
    return rows


def normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def imported_distributions(repo: Path) -> list[str]:
    """Conservatively map repository-wide top-level imports to distributions."""

    local_roots = {path.stem for path in repo.glob("*.py")}
    local_roots.update(path.name for path in repo.iterdir() if path.is_dir())
    module_to_distribution = {
        "lm_eval": "lm-eval",
        "numpy": "numpy",
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "tqdm": "tqdm",
        "matplotlib": "matplotlib",
        "termcolor": "termcolor",
        "accelerate": "accelerate",
        "bitblas": "bitblas",
        "scipy": "scipy",
    }
    found: set[str] = set()
    for path in repo.rglob("*.py"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(_read_text(path), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            root: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    candidate = alias.name.split(".", 1)[0]
                    if candidate in module_to_distribution:
                        found.add(module_to_distribution[candidate])
                continue
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".", 1)[0]
            if root and root not in local_roots and root in module_to_distribution:
                found.add(module_to_distribution[root])
    return sorted(found, key=normalize_distribution)


def missing_local_imports(repo: Path) -> list[dict[str, Any]]:
    """Find unresolved imports whose names look repository-local."""

    local_roots = {path.stem for path in repo.glob("*.py")}
    local_roots.update(path.name for path in repo.iterdir() if path.is_dir())
    likely_local_prefixes = ("datautils", "e2e_", "trans_utils", "utils")
    findings: list[dict[str, Any]] = []
    for path in repo.rglob("*.py"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(_read_text(path), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 0 or not node.module:
                continue
            root = node.module.split(".", 1)[0]
            if root in local_roots or not root.startswith(likely_local_prefixes):
                continue
            findings.append(
                {
                    "file": path.relative_to(repo).as_posix(),
                    "line": node.lineno,
                    "module": node.module,
                }
            )
    return findings


ABSOLUTE_PATH_RE = re.compile(
    r"(?P<quote>['\"])(?P<path>/(?:home|mnt|data(?:\d+)?|opt)/[^'\"\n]*?)(?P=quote)"
)


def absolute_path_literals(repo: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in repo.rglob("*.py"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        text = _read_text(path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in ABSOLUTE_PATH_RE.finditer(line):
                literal = match.group("path")
                records.append(
                    {
                        "file": path.relative_to(repo).as_posix(),
                        "line": line_number,
                        "path": literal,
                        "exists_on_audit_host": Path(literal).exists(),
                    }
                )
    unique: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        unique[(record["file"], record["line"], record["path"])] = record
    return list(unique.values())


def inspect_model_cache(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False}
    result: dict[str, Any] = {
        "provided": True,
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return result
    snapshot = path
    revision: str | None = None
    ref = path / "refs" / "main"
    if ref.exists():
        revision = _read_text(ref).strip()
        snapshot = path / "snapshots" / revision
    result.update({"revision": revision, "snapshot": str(snapshot), "snapshot_exists": snapshot.exists()})
    index_path = snapshot / "model.safetensors.index.json"
    referenced_shards: list[str] = []
    invalid_index: str | None = None
    if index_path.exists():
        try:
            index = json.loads(_read_text(index_path))
            referenced_shards = sorted(set(index.get("weight_map", {}).values()))
        except (json.JSONDecodeError, AttributeError) as exc:
            invalid_index = f"{type(exc).__name__}: {exc}"
    missing_shards = [name for name in referenced_shards if not (snapshot / name).exists()]
    essential = ["config.json", "tokenizer_config.json"]
    missing_essential = [name for name in essential if not (snapshot / name).exists()]
    shard_checks: list[dict[str, Any]] = []
    for name in referenced_shards:
        shard = snapshot / name
        check: dict[str, Any] = {
            "name": name,
            "exists": shard.exists(),
            "size_bytes": shard.stat().st_size if shard.exists() else None,
            "nonzero": shard.exists() and shard.stat().st_size > 8,
            "safetensors_header_json_valid": False,
        }
        if shard.exists() and shard.stat().st_size > 8:
            try:
                with shard.open("rb") as handle:
                    header_size = int.from_bytes(handle.read(8), "little")
                    check["header_size_bytes"] = header_size
                    if 0 < header_size <= min(64 * 1024 * 1024, shard.stat().st_size - 8):
                        header = json.loads(handle.read(header_size).decode("utf-8"))
                        check["safetensors_header_json_valid"] = isinstance(header, dict)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                check["header_error"] = f"{type(exc).__name__}: {exc}"
        shard_checks.append(check)
    tokenizer_candidates = ["tokenizer.json", "tokenizer.model", "vocab.json"]
    tokenizer_assets = [name for name in tokenizer_candidates if (snapshot / name).exists()]
    minimal_files_present = (
        bool(referenced_shards)
        and not invalid_index
        and not missing_shards
        and not missing_essential
        and bool(tokenizer_assets)
        and all(
            check["nonzero"] and check["safetensors_header_json_valid"]
            for check in shard_checks
        )
    )
    result.update(
        {
            "index_present": index_path.exists(),
            "index_error": invalid_index,
            "referenced_shards": referenced_shards,
            "missing_referenced_shards": missing_shards,
            "missing_essential_files": missing_essential,
            "shard_checks": shard_checks,
            "tokenizer_assets_present": tokenizer_assets,
            "minimal_files_present": minimal_files_present,
            "appears_complete": minimal_files_present,
            "completeness_claim": (
                "minimal local-file/header gate only; no full shard hash, tensor load, or model forward"
            ),
        }
    )
    return result


def inspect_path_probes(probes: Iterable[tuple[str, Path]]) -> list[dict[str, Any]]:
    return [{"name": name, "path": str(path), "exists": path.exists()} for name, path in probes]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_patch(
    patch_file: Path | None,
    *,
    clean_repo: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Check a compatibility patch against the clean tree without applying it."""

    if patch_file is None:
        return {"provided": False}
    patch_file = patch_file.resolve()
    result: dict[str, Any] = {
        "provided": True,
        "path": str(patch_file),
        "logical_path": "external_patches/liftquant_72b3875_qwen2_attention_type.patch",
        "exists": patch_file.exists(),
    }
    if not patch_file.exists():
        return result
    text = _read_text(patch_file)
    declared_modes = sorted(set(re.findall(r"\b100[0-7]{3}\b", text)))
    target = clean_repo / "quantize" / "liftq.py"
    target_mode_probe = _run_limited(
        ["git", "ls-files", "--stage", "--", "quantize/liftq.py"],
        cwd=clean_repo,
        timeout_seconds=timeout_seconds,
    )
    mode_match = re.match(r"(100[0-7]{3})\s", target_mode_probe["stdout_excerpt"])
    clean_target_git_mode = mode_match.group(1) if mode_match else None
    result.update(
        {
            "size_bytes": patch_file.stat().st_size,
            "sha256": _sha256_file(patch_file),
            "declared_file_modes": declared_modes,
            "clean_target_git_mode": clean_target_git_mode,
            "clean_target_filesystem_mode": (
                f"{target.stat().st_mode & 0o777:o}" if target.exists() else None
            ),
            "target_mode_probe": target_mode_probe,
            "hunk_count": len(re.findall(r"^@@ ", text, flags=re.MULTILINE)),
            "preserves_attention_type": "self.attention_type = module.attention_type" in text,
        }
    )
    result["mode_metadata_matches_clean_target"] = (
        bool(clean_target_git_mode)
        and (not declared_modes or clean_target_git_mode in declared_modes)
    )
    result["git_apply_check"] = _run_limited(
        ["git", "apply", "--check", str(patch_file)],
        cwd=clean_repo,
        timeout_seconds=timeout_seconds,
    )
    return result


def _parse_main_argv(stdout: str) -> list[str]:
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate.startswith("['main.py'"):
            continue
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed
    return []


def _argv_options(argv: Sequence[str]) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    index = 1
    while index < len(argv):
        item = argv[index]
        if not item.startswith("--"):
            index += 1
            continue
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            options[item] = argv[index + 1]
            index += 2
        else:
            options[item] = True
            index += 1
    return options


def inspect_smoke_run(
    run_dir: Path | None,
    *,
    source_variant: str,
    physical_gpu: str | None = None,
) -> dict[str, Any]:
    """Summarize already-finished smoke artifacts; never start or resume a job."""

    if run_dir is None:
        return {"provided": False, "source_variant": source_variant}
    run_dir = run_dir.resolve()
    result: dict[str, Any] = {
        "provided": True,
        "source_variant": source_variant,
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "physical_gpu": physical_gpu,
    }
    if not run_dir.exists():
        return result

    def small_text(name: str, limit: int = 8_000) -> str:
        path = run_dir / name
        return _read_text(path)[:limit] if path.exists() else ""

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    time_path = run_dir / "time.txt"
    exit_path = run_dir / "exit_code"
    status_path = run_dir / "status"
    stdout = _read_text(stdout_path) if stdout_path.exists() else ""
    stderr = _read_text(stderr_path) if stderr_path.exists() else ""
    argv = _parse_main_argv(stdout)
    options = _argv_options(argv)
    exit_text = small_text("exit_code", 64).strip()
    try:
        exit_code: int | None = int(exit_text)
    except ValueError:
        exit_code = None
    time_text = small_text("time.txt", 512).strip()
    elapsed_match = re.search(r"elapsed=([0-9.]+)", time_text)
    rss_match = re.search(r"maxrss_kb=([0-9]+)", time_text)
    error_lines = [
        line.strip()
        for line in stderr.splitlines()
        if re.search(r"(?:Error|Exception):", line)
    ]
    artifacts: list[dict[str, Any]] = []
    for artifact in sorted(run_dir.rglob("*.pth")):
        record: dict[str, Any] = {
            "path": str(artifact),
            "relative_path": artifact.relative_to(run_dir).as_posix(),
            "size_bytes": artifact.stat().st_size,
        }
        if artifact.stat().st_size <= 64 * 1024 * 1024:
            record["sha256"] = _sha256_file(artifact)
        artifacts.append(record)
    result.update(
        {
            "exit_code": exit_code,
            "status_file": small_text("status", 128).strip() or None,
            "elapsed_seconds": float(elapsed_match.group(1)) if elapsed_match else None,
            "maxrss_kb": int(rss_match.group(1)) if rss_match else None,
            "argv": argv,
            "resolved_options": options,
            "error_signature": error_lines[-1] if error_lines else None,
            "success_markers": {
                "started_layer0": "=== Start quantize layer 0 ===" in stdout,
                "reported_layer0_save": "20to8-layer0.pth" in stdout,
                "reported_main_elapsed": bool(
                    re.search(r"\(main\.py \d+\): INFO [0-9.]+", stdout)
                ),
            },
            "stdout_sha256": _sha256_file(stdout_path) if stdout_path.exists() else None,
            "stderr_sha256": _sha256_file(stderr_path) if stderr_path.exists() else None,
            "time_sha256": _sha256_file(time_path) if time_path.exists() else None,
            "exit_code_sha256": _sha256_file(exit_path) if exit_path.exists() else None,
            "status_sha256": _sha256_file(status_path) if status_path.exists() else None,
            "stdout_tail": stdout[-2_000:],
            "stderr_tail": stderr[-2_000:],
            "artifacts": artifacts,
            "scope": {
                "model": options.get("--net"),
                "mapping": options.get("--expc"),
                "wbits": options.get("--wbits"),
                "seqlen": options.get("--seqlen"),
                "nsamples1": options.get("--nsamples1"),
                "nsamples2": options.get("--nsamples2"),
                "epochs1": options.get("--epochs1"),
                "epochs2": options.get("--epochs2"),
                "batch_size": options.get("--batch_size"),
                "quant_start": options.get("--quant_start"),
                "quant_end": options.get("--quant_end"),
                "eval_ppl": bool(options.get("--eval_ppl", False)),
                "tasks": options.get("--tasks", ""),
            },
        }
    )
    return result


def smoke_run_completion_gate(smoke: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless an already-finished patched smoke has complete evidence."""

    markers = smoke.get("success_markers") or {}
    layer0_artifacts = [
        artifact
        for artifact in smoke.get("artifacts", [])
        if str(artifact.get("relative_path", "")).endswith("-layer0.pth")
    ]
    checks = {
        "provided_and_exists": bool(smoke.get("provided") and smoke.get("exists")),
        "exit_code_zero": smoke.get("exit_code") == 0,
        "status_completed": smoke.get("status_file") == "COMPLETED",
        "all_success_markers": bool(markers) and all(markers.values()),
        "nonempty_hashed_layer0_artifact": bool(layer0_artifacts)
        and all(
            int(artifact.get("size_bytes", 0)) > 0
            and bool(re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", ""))))
            for artifact in layer0_artifacts
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _installed_versions(
    python: Path,
    names: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    probe = (
        "import importlib.metadata as m,json,platform,sys;"
        f"names={json.dumps(list(names))};"
        "out={'python':platform.python_version(),'executable':sys.executable,'packages':{}};"
        "[(out['packages'].__setitem__(n,(m.version(n) if _has(n,m) else None))) for n in names];"
        "print(json.dumps(out,sort_keys=True))"
    )
    # Define the helper in the expression-only script without importing the repo.
    probe = "def _has(n,m):\n try:\n  m.version(n); return True\n except m.PackageNotFoundError:\n  return False\n" + probe
    evidence = _run_limited(
        [str(python), "-B", "-I", "-c", probe],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    parsed: dict[str, Any] | None = None
    if evidence["returncode"] == 0:
        try:
            parsed = json.loads(evidence["stdout_excerpt"].strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            parsed = None
    return {"probe": evidence, "metadata": parsed}


def _help_probe(python: Path, script: Path, repo: Path, timeout_seconds: int) -> dict[str, Any]:
    if not script.exists():
        return {
            "command": [str(python), "-B", script.name, "--help"],
            "returncode": None,
            "timed_out": False,
            "status": "missing_script",
            "runtime_flags": [],
            "stdout_excerpt": "",
            "stderr_excerpt": f"missing script: {script}",
        }
    result = _run_limited(
        [str(python), "-B", script.name, "--help"],
        cwd=repo,
        timeout_seconds=timeout_seconds,
    )
    combined = result["stdout_excerpt"] + "\n" + result["stderr_excerpt"]
    result["runtime_flags"] = sorted(set(re.findall(r"--[A-Za-z0-9_-]+", combined)))
    result["status"] = "ok" if result["returncode"] == 0 else "failed"
    return result


def _git_info(repo: Path, expected_commit: str | None, timeout_seconds: int) -> dict[str, Any]:
    head_probe = _run_limited(
        ["git", "rev-parse", "HEAD"], cwd=repo, timeout_seconds=timeout_seconds
    )
    status_probe = _run_limited(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=repo,
        timeout_seconds=timeout_seconds,
    )
    source_status_probe = _run_limited(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            ".",
            ":(exclude)**/__pycache__/**",
            ":(exclude)**/*.pyc",
        ],
        cwd=repo,
        timeout_seconds=timeout_seconds,
    )
    head = head_probe["stdout_excerpt"].strip().splitlines()
    head_ok = (
        head_probe["returncode"] == 0
        and not head_probe["timed_out"]
        and bool(head)
        and bool(re.fullmatch(r"[0-9a-fA-F]{40}", head[-1]))
    )
    status_ok = status_probe["returncode"] == 0 and not status_probe["timed_out"]
    source_status_ok = (
        source_status_probe["returncode"] == 0 and not source_status_probe["timed_out"]
    )
    provenance_verified = head_ok and status_ok and source_status_ok
    commit = head[-1].lower() if head_ok else None
    dirty_entries = [line for line in status_probe["stdout_excerpt"].splitlines() if line.strip()]
    source_dirty = [
        line for line in source_status_probe["stdout_excerpt"].splitlines() if line.strip()
    ]
    return {
        "commit": commit,
        "expected_commit": expected_commit,
        "provenance_verified": provenance_verified,
        "commit_matches": head_ok
        and (expected_commit is None or commit == expected_commit.lower()),
        "worktree_clean": status_ok and not status_probe["stdout_nonempty"],
        "source_worktree_clean": source_status_ok
        and not source_status_probe["stdout_nonempty"],
        "dirty_entries": dirty_entries,
        "dirty_entry_count": status_probe["stdout_line_count"] if status_ok else None,
        "dirty_entries_truncated": status_probe["stdout_truncated"],
        "source_dirty_entries": source_dirty,
        "source_dirty_entry_count": (
            source_status_probe["stdout_line_count"] if source_status_ok else None
        ),
        "source_dirty_entries_truncated": source_status_probe["stdout_truncated"],
        "head_probe": head_probe,
        "status_probe": status_probe,
        "source_status_probe": source_status_probe,
    }


def _requirement_audit(
    requirements: list[dict[str, str | None]], installed: dict[str, Any]
) -> list[dict[str, Any]]:
    metadata = installed.get("metadata") or {}
    packages = metadata.get("packages", {})
    rows: list[dict[str, Any]] = []
    for requirement in requirements:
        name = str(requirement["name"])
        version = packages.get(name)
        if version is None:
            status = "missing"
        elif requirement["exact_pin"] is None:
            status = "present_unpinned"
        elif version == requirement["exact_pin"]:
            status = "exact_match"
        else:
            status = "version_mismatch"
        rows.append({**requirement, "installed": version, "status": status})
    return rows


def _sample_semantics(repo: Path) -> dict[str, Any]:
    main = _read_text(repo / "main.py") if (repo / "main.py").exists() else ""
    liftq = _read_text(repo / "quantize" / "liftq.py") if (repo / "quantize" / "liftq.py").exists() else ""
    return {
        "main_uses_max_of_two_sample_flags": "args.nsamples = max(args.nsamples1, args.nsamples2)" in main,
        "stage1_reduces_equal_max_by_one_thirty_second": bool(
            re.search(r"args\.nsamples1\s*=\s*args\.nsamples1\s*-\s*args\.nsamples\s*//\s*32", liftq)
        ),
        "stage2_reduces_equal_max_by_one_thirty_second": bool(
            re.search(r"args\.nsamples2\s*=\s*args\.nsamples2\s*-\s*args\.nsamples2\s*//\s*32", liftq)
        ),
        "implication_for_4096_4096": (
            "Both stages iterate over 3968 samples after the in-code 1/32 reduction; "
            "the loaded calibration pool remains 4096."
        ),
    }


def _command_templates(model_cache: dict[str, Any]) -> dict[str, Any]:
    snapshot = model_cache.get("snapshot") or "/path/to/Qwen2.5-3B-Instruct"
    smoke = f"""PYTHONDONTWRITEBYTECODE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \\
/home/wangmeiqi/codex_envs/liftquant-72b3875/bin/python main.py \\
  --model {snapshot} \\
  --net Qwen2.5-3B-Instruct \\
  --cache_dir /home/wangmeiqi/codex_scratch/liftquant-smoke/cache \\
  --output_dir /home/wangmeiqi/codex_scratch/liftquant-smoke/log \\
  --save_dir /home/wangmeiqi/codex_scratch/liftquant-smoke/qmodels \\
  --wbits 2 --expc 20to8 --w_sym --abits 16 --kbits 16 --vbits 16 \\
  --calib_dataset wikitext2 --seqlen 128 \\
  --nsamples1 2 --nsamples2 2 --epochs1 0 --epochs2 0 --batch_size 1 \\
  --quant_start 0 --quant_end 1 --training_trans --usefullfp --limit 1"""
    full = f"""CUDA_VISIBLE_DEVICES=0 /home/wangmeiqi/codex_envs/liftquant-72b3875/bin/python main.py \\
  --model {snapshot} --net Qwen2.5-3B-Instruct \\
  --save_dir /path/to/qmodels --output_dir /path/to/log --cache_dir /path/to/cache \\
  --eval_ppl --wbits 2 --expc 20to8 --w_sym \\
  --abits 16 --kbits 16 --vbits 16 --true-sequential --act-order --use_fpinps \\
  --Rres_init Hadamard --nsamples1 4096 --nsamples2 4096 \\
  --epochs1 2 --epochs2 2 --batch_size 2 --calib_dataset redpajama \\
  --usefullfp --training_trans --finetuning_weights --align 1 \\
  --lscale_lr 5e-3 --lexw_lr 2e-2 --lw_lr 2e-5 --la_lr 2e-3 --lt_lr 2e-4"""
    return {
        "reduced_qwen_smoke": {
            "status": "template_only_not_executed",
            "purpose": "model-load, cached WikiText, lifting initialization, and one-layer control-flow smoke",
            "limitations": (
                "该模板把两个 optimization epoch 都设为 0，且没有 endpoint metric；"
                "只能做更小的控制流排障，不能复现 LiftQuant 精度。实际执行时仍需要 CUDA。"
            ),
            "command": smoke,
        },
        "block_correction_full_after_path_fix": {
            "status": "blocked_template_not_executed",
            "blocking_conditions": [
                "replace the hard-coded RedPajama cache with a reachable/configurable path",
                "confirm whether README --epochs 2 means two epochs for each code stage or two epochs total",
                "record actual GPU-hours, peak memory, calibration tokens, artifact bytes, and endpoint metrics",
            ],
            "mapping_assumption": {
                "--nsamples 4096": "--nsamples1 4096 --nsamples2 4096",
                "--epochs 2": "--epochs1 2 --epochs2 2",
            },
            "command": full,
        },
    }


def build_audit(
    repo: Path,
    *,
    python: Path,
    expected_commit: str | None = EXPECTED_COMMIT,
    model_cache: Path | None = None,
    path_probes: Iterable[tuple[str, Path]] = (),
    patch_file: Path | None = None,
    official_smoke_run: Path | None = None,
    patched_smoke_run: Path | None = None,
    patched_smoke_gpu: str | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Build the complete JSON-serializable audit record."""

    repo = repo.resolve()
    readme = _read_text(repo / "README.md") if (repo / "README.md").exists() else ""
    commands = extract_readme_commands(readme)
    main_records = argparse_flags_from_ast(repo / "main.py")
    main_flags = sorted({flag for record in main_records for flag in record["flags"]})
    e2e_records = dataclass_flags_from_ast(repo / "e2efinetune.py")
    e2e_local_flags = sorted({record["flag"] for record in e2e_records})
    main_command = next((command for command in commands if command["entrypoint"] == "main.py"), None)
    e2e_command = next((command for command in commands if command["entrypoint"] == "e2efinetune.py"), None)
    main_readme_flags = main_command["flags"] if main_command else []
    e2e_readme_flags = e2e_command["flags"] if e2e_command else []

    requirements = parse_requirements(repo / "requirements.txt")
    required_names = [str(row["name"]) for row in requirements]
    installed = _installed_versions(
        python, required_names, cwd=repo, timeout_seconds=timeout_seconds
    )
    requirement_rows = _requirement_audit(requirements, installed)
    imported = imported_distributions(repo)
    declared = {normalize_distribution(str(row["name"])) for row in requirements}
    missing_declarations = [name for name in imported if normalize_distribution(name) not in declared]
    absolute_paths = absolute_path_literals(repo)
    hardcoded_redpajama = [
        record
        for record in absolute_paths
        if "redpajama" in record["path"].lower() or "shared_data/datasets" in record["path"]
    ]
    model_info = inspect_model_cache(model_cache)
    git = _git_info(repo, expected_commit, timeout_seconds)
    patch_info = inspect_patch(
        patch_file, clean_repo=repo, timeout_seconds=timeout_seconds
    )
    official_smoke = inspect_smoke_run(
        official_smoke_run,
        source_variant="official_pinned_commit_unpatched",
    )
    patched_smoke = inspect_smoke_run(
        patched_smoke_run,
        source_variant="official_pinned_commit_plus_compatibility_patch",
        physical_gpu=patched_smoke_gpu,
    )
    patched_smoke_gate = smoke_run_completion_gate(patched_smoke)
    main_help = _help_probe(python, repo / "main.py", repo, timeout_seconds)
    e2e_help = _help_probe(python, repo / "e2efinetune.py", repo, timeout_seconds)
    missing_imports = missing_local_imports(repo)
    e2e_source = _read_text(repo / "e2efinetune.py") if (repo / "e2efinetune.py").exists() else ""
    remaining_strings = "return_remaining_strings=True" in e2e_source
    extra_args_occurrences = len(re.findall(r"\bextra_args\b", e2e_source))

    findings: list[dict[str, str]] = []
    missing_main_static = sorted(set(main_readme_flags) - set(main_flags))
    missing_main_runtime = sorted(
        set(main_readme_flags) - set(main_help.get("runtime_flags", []))
    )
    unsupported_main = (
        sorted(set(missing_main_static) & set(missing_main_runtime))
        if main_help["status"] == "ok"
        else []
    )
    if unsupported_main:
        findings.append(
            {
                "id": "README_MAIN_FLAGS_NOT_IMPLEMENTED",
                "severity": "HIGH",
                "scope": "block_correction",
                "evidence": (
                    f"README uses {unsupported_main}; they are absent from both main.py AST "
                    "and the successful runtime --help output."
                ),
                "impact": "The published block-correction command exits in argparse before model loading.",
                "action": "Use explicit stage flags and ask upstream to define the intended two-stage mapping.",
            }
        )
    missing_redpajama = [record for record in hardcoded_redpajama if not record["exists_on_audit_host"]]
    if missing_redpajama:
        findings.append(
            {
                "id": "REDPAJAMA_CACHE_NOT_PORTABLE",
                "severity": "HIGH",
                "scope": "block_and_e2e",
                "evidence": "; ".join(
                    f"{record['file']}:{record['line']} -> {record['path']} (missing)"
                    for record in missing_redpajama
                ),
                "impact": "README RedPajama calibration/fine-tuning cannot start on the audited 210 host.",
                "action": "Make dataset cache/config a CLI argument and verify the exact dataset revision/split.",
            }
        )
    if main_help["status"] != "ok":
        findings.append(
            {
                "id": "MAIN_HELP_PROBE_FAILED",
                "severity": "HIGH",
                "scope": "block_correction",
                "evidence": (
                    f"main.py --help status={main_help['status']} "
                    f"returncode={main_help.get('returncode')} timed_out={main_help.get('timed_out')}"
                ),
                "impact": "The block entrypoint cannot pass the minimum CPU-only import/parser gate.",
                "action": "Repair the environment or entrypoint before scheduling any GPU smoke.",
            }
        )
    if not git["provenance_verified"]:
        findings.append(
            {
                "id": "GIT_PROVENANCE_PROBE_FAILED",
                "severity": "HIGH",
                "scope": "provenance",
                "evidence": (
                    f"head_rc={git['head_probe']['returncode']}, "
                    f"status_rc={git['status_probe']['returncode']}, "
                    f"source_status_rc={git['source_status_probe']['returncode']}"
                ),
                "impact": "Commit identity and clean-source status are unverified and must fail closed.",
                "action": "Use a valid Git checkout and rerun all three bounded provenance probes.",
            }
        )
    if e2e_help["status"] != "ok":
        findings.append(
            {
                "id": "E2E_ENTRYPOINT_IMPORT_FAILURE",
                "severity": "HIGH",
                "scope": "e2e_finetuning",
                "evidence": e2e_help["stderr_excerpt"].strip().splitlines()[-1] if e2e_help["stderr_excerpt"].strip() else "--help failed",
                "impact": "The optional E2E path cannot reach argument parsing, even for --help.",
                "action": "Restore/provide datautils_block.py (or remove the import) and rerun the help-only gate.",
            }
        )
    drift = [row for row in requirement_rows if row["status"] in {"missing", "version_mismatch"}]
    if drift:
        findings.append(
            {
                "id": "ENVIRONMENT_REQUIREMENT_DRIFT",
                "severity": "MEDIUM",
                "scope": "environment",
                "evidence": "; ".join(
                    f"{row['name']} declared={row['exact_pin'] or 'unpinned'} installed={row['installed']}"
                    for row in drift
                ),
                "impact": "A help-capable environment is not evidence of a requirements-faithful training environment.",
                "action": "Build a Python 3.12 environment from a satisfiable lock and capture pip freeze plus CUDA ABI.",
            }
        )
    if missing_declarations:
        findings.append(
            {
                "id": "DIRECT_IMPORTS_NOT_DECLARED",
                "severity": "MEDIUM",
                "scope": "environment",
                "evidence": f"Repository imports not listed in requirements.txt: {missing_declarations}.",
                "impact": "A clean install can fail or depend on accidental transitive packages.",
                "action": "Declare and pin direct runtime dependencies for each block/E2E/chat extra.",
            }
        )
    if missing_imports:
        findings.append(
            {
                "id": "MISSING_REPOSITORY_LOCAL_IMPORT",
                "severity": "HIGH",
                "scope": "e2e_finetuning",
                "evidence": "; ".join(
                    f"{row['file']}:{row['line']} imports {row['module']}" for row in missing_imports
                ),
                "impact": "The affected entrypoint is incomplete at the pinned commit.",
                "action": "Obtain the missing source from upstream with provenance; do not invent benchmark logic locally.",
            }
        )
    sample_semantics = _sample_semantics(repo)
    if sample_semantics["stage1_reduces_equal_max_by_one_thirty_second"]:
        findings.append(
            {
                "id": "CALIBRATION_POOL_VS_OPTIMIZATION_COUNT",
                "severity": "MEDIUM",
                "scope": "block_correction",
                "evidence": sample_semantics["implication_for_4096_4096"],
                "impact": "Loaded-pool size and per-stage optimization-example count must not be reported as the same quantity.",
                "action": "Log both counts and compare methods by actual tokens consumed by each optimization stage.",
            }
        )
    if remaining_strings and extra_args_occurrences <= 1:
        findings.append(
            {
                "id": "E2E_UNKNOWN_FLAGS_CAN_BE_SILENT",
                "severity": "MEDIUM",
                "scope": "e2e_finetuning",
                "evidence": "HfArgumentParser returns remaining strings into extra_args, which is not consumed after assignment.",
                "impact": "Misspelled or version-removed E2E flags can be silently ignored once imports are repaired.",
                "action": "Fail if extra_args is non-empty and persist the resolved TrainingArguments object.",
            }
        )
    if official_smoke.get("provided") and official_smoke.get("exit_code") != 0:
        findings.append(
            {
                "id": "OFFICIAL_QWEN_SMOKE_COMPATIBILITY_FAILURE",
                "severity": "HIGH",
                "scope": "block_correction_qwen_smoke",
                "evidence": str(official_smoke.get("error_signature")),
                "impact": "The unmodified pinned commit cannot complete the Qwen2.5 layer-0 smoke in the audited Transformers 4.57 environment.",
                "action": "Keep the official failure as the baseline and report any compatibility patch as a separate source variant.",
            }
        )
    if patched_smoke.get("provided") and patched_smoke.get("exit_code") == 0:
        findings.append(
            {
                "id": "PATCHED_LAYER0_SMOKE_IS_NOT_FULL_REPRODUCTION",
                "severity": "MEDIUM",
                "scope": "block_correction_qwen_smoke",
                "evidence": (
                    f"patched layer-0 exit=0, elapsed={patched_smoke.get('elapsed_seconds')}s, "
                    f"scope={patched_smoke.get('scope')}"
                ),
                "impact": "This validates a bounded execution path and artifact write, not full-model PPL, task accuracy, or deployment bytes.",
                "action": "Retain external_reproduction_pending and run the preregistered full endpoint separately.",
            }
        )
    if patch_info.get("provided") and not patch_info.get("mode_metadata_matches_clean_target", True):
        findings.append(
            {
                "id": "COMPATIBILITY_PATCH_MODE_METADATA_MISMATCH",
                "severity": "LOW",
                "scope": "compatibility_patch",
                "evidence": (
                    f"patch modes={patch_info.get('declared_file_modes')} vs clean target "
                    f"{patch_info.get('clean_target_git_mode')}"
                ),
                "impact": "Patch application can warn about file-mode provenance even if the content hunk applies.",
                "action": "Regenerate the patch with the official target mode before publishing it.",
            }
        )
    if not git["worktree_clean"] and git["source_worktree_clean"]:
        findings.append(
            {
                "id": "BYTECODE_ONLY_WORKTREE_DRIFT",
                "severity": "LOW",
                "scope": "provenance",
                "evidence": f"git status contains {len(git['dirty_entries'])} bytecode/cache entries and no source entry.",
                "impact": "Commit identity is intact, but a clean-checkout claim would be inaccurate.",
                "action": "Use a fresh read-only checkout or PYTHONDONTWRITEBYTECODE=1 for the actual reproduction.",
            }
        )
    if not git["source_worktree_clean"]:
        findings.append(
            {
                "id": "SOURCE_WORKTREE_DIFFERS_FROM_PINNED_COMMIT",
                "severity": "HIGH",
                "scope": "provenance",
                "evidence": "; ".join(git["source_dirty_entries"]),
                "impact": "Static findings from this path may describe a compatibility adapter rather than the official pinned source.",
                "action": "Audit a clean local clone/export of the pinned commit and report patched smoke results separately.",
            }
        )

    common_gate_failed = bool(
        not git["provenance_verified"]
        or not git["commit_matches"]
        or not git["source_worktree_clean"]
    )
    block_blocked = bool(
        common_gate_failed
        or main_help["status"] != "ok"
        or unsupported_main
        or missing_redpajama
    )
    e2e_blocked = bool(
        common_gate_failed
        or e2e_help["status"] != "ok"
        or missing_imports
        or missing_redpajama
    )
    return {
        "schema_version": 2,
        "verified_as_of": VERIFIED_AS_OF,
        "audit_scope": (
            "audit-generation process only (static/runtime compatibility probes); "
            "external official-unpatched and compatibility-patched smoke evidence is inspected separately"
        ),
        "repository": str(repo),
        "provenance": git,
        "entrypoints": {
            "main": {
                "static_parser_records": main_records,
                "static_flags": main_flags,
                "readme_flags": main_readme_flags,
                "readme_flags_missing_from_static_parser": missing_main_static,
                "readme_flags_missing_from_runtime_help": missing_main_runtime,
                "unsupported_readme_flags": unsupported_main,
                "help_probe": main_help,
            },
            "e2e": {
                "local_dataclass_records": e2e_records,
                "local_dataclass_flags": e2e_local_flags,
                "readme_flags": e2e_readme_flags,
                "readme_flags_not_in_local_dataclasses": sorted(set(e2e_readme_flags) - set(e2e_local_flags)),
                "inherits_transformers_training_arguments": any(
                    "Seq2SeqTrainingArguments" in base
                    for record in e2e_records
                    for base in record["class_bases"]
                ),
                "return_remaining_strings": remaining_strings,
                "extra_args_occurrences": extra_args_occurrences,
                "help_probe": e2e_help,
                "missing_local_imports": missing_imports,
            },
        },
        "readme_commands": commands,
        "requirements": {
            "declared": requirements,
            "installed_probe": installed,
            "comparison": requirement_rows,
            "repository_imported_distributions": imported,
            "imported_but_not_declared": missing_declarations,
            "readme_python_version": "3.12" if "python=3.12" in readme else None,
        },
        "paths": {
            "absolute_literals": absolute_paths,
            "hardcoded_redpajama": hardcoded_redpajama,
            "additional_probes": inspect_path_probes(path_probes),
        },
        "model_cache": model_info,
        "patched_smoke_evidence": {
            "separation_rule": (
                "干净 official commit 是兼容性基线；patched run 只是本地 adapter smoke，"
                "不得重新标成官方完整复现。"
            ),
            "compatibility_patch": patch_info,
            "official_unpatched_run": official_smoke,
            "patched_run": patched_smoke,
            "patched_run_completion_gate": patched_smoke_gate,
        },
        "sample_accounting": sample_semantics,
        "compatibility": {
            "block_correction": "blocked_pending_upstream_or_local_adapter" if block_blocked else "help_gate_ready_only",
            "optional_e2e": "blocked_at_import" if e2e_blocked else "help_gate_ready_only",
            "external_reproduction": "pending",
            "accuracy_claim_allowed": False,
            "patched_layer0_smoke": (
                "control_flow_and_artifact_passed"
                if patched_smoke_gate["passed"]
                else "not_passed_or_not_provided"
            ),
        },
        "commands": _command_templates(model_info),
        "findings": findings,
        "execution": {
            "scope": "audit_process_only",
            "does_not_describe_external_smoke_evidence": True,
            "training_executed": False,
            "quantization_executed": False,
            "gpu_job_executed": False,
            "model_loaded": False,
            "dataset_loaded": False,
            "subprocess_policy": "git metadata, importlib.metadata, and --help only; shell=False; bounded timeout",
        },
    }


def render_summary(audit: dict[str, Any]) -> str:
    provenance = audit["provenance"]
    main = audit["entrypoints"]["main"]
    e2e = audit["entrypoints"]["e2e"]
    req = audit["requirements"]
    model = audit["model_cache"]
    smoke = audit["patched_smoke_evidence"]
    patch = smoke["compatibility_patch"]
    official_smoke = smoke["official_unpatched_run"]
    patched_smoke = smoke["patched_run"]
    layer0_artifact = next(
        (
            artifact
            for artifact in patched_smoke.get("artifacts", [])
            if artifact["relative_path"].endswith("20to8-layer0.pth")
        ),
        None,
    )
    if provenance["worktree_clean"]:
        provenance_note = "- 本次静态证据来自干净副本；compatibility-patched smoke 在下节单列。"
    elif provenance["source_worktree_clean"]:
        provenance_note = (
            f"- 当前 dirty 条目数：{len(provenance['dirty_entries'])}，仅为 Python bytecode/cache；"
            "正式复现仍应使用干净只读 checkout。"
        )
    else:
        provenance_note = (
            "- 检测到源文件修改：`"
            + "`, `".join(provenance["source_dirty_entries"])
            + "`；该路径不能代表未修改的官方 commit。"
        )
    smoke_lines: list[str] = []
    if official_smoke.get("provided") or patched_smoke.get("provided"):
        smoke_lines = [
            "## Official 与 compatibility-patched smoke 的边界",
            "",
            smoke["separation_rule"],
            "",
            f"- **Official pinned source（未打补丁）**：exit=`{official_smoke.get('exit_code')}`，"
            f"elapsed=`{official_smoke.get('elapsed_seconds')}` s，max RSS=`{official_smoke.get('maxrss_kb')}` KiB；"
            f"失败签名：`{official_smoke.get('error_signature')}`。",
            f"- **兼容补丁**：`{patch.get('logical_path')}`，SHA256=`{patch.get('sha256')}`，"
            f"`git apply --check` 返回码=`{(patch.get('git_apply_check') or {}).get('returncode')}`，"
            f"patch mode 与 clean target 一致=`{patch.get('mode_metadata_matches_clean_target')}`。",
            f"- **Pinned source + compatibility patch**：物理 GPU=`{patched_smoke.get('physical_gpu')}`，"
            f"exit=`{patched_smoke.get('exit_code')}`，elapsed=`{patched_smoke.get('elapsed_seconds')}` s，"
            f"max RSS=`{patched_smoke.get('maxrss_kb')}` KiB，status=`{patched_smoke.get('status_file')}`。",
            f"- 成功运行证据 SHA256：stdout=`{patched_smoke.get('stdout_sha256')}`；"
            f"stderr=`{patched_smoke.get('stderr_sha256')}`；time=`{patched_smoke.get('time_sha256')}`。",
            f"- Layer-0 artifact：`{(layer0_artifact or {}).get('path')}`；bytes=`{(layer0_artifact or {}).get('size_bytes')}`；"
            f"SHA256=`{(layer0_artifact or {}).get('sha256')}`。",
            "",
            "成功 smoke 的 resolved scope 是 Qwen2.5-3B、`20to8`/`wbits=2`、WikiText2、"
            "`seqlen=128`、`nsamples1=8`、`nsamples2=8`、`epochs1=1`、`epochs2=1`、batch=2、"
            "`quant_start=0`、`quant_end=1`。两阶段复用由 `max(nsamples1, nsamples2)` 加载的 8-window 池；"
            "不能写成 16 个独立校准窗口。运行未设置 `--eval_ppl`，tasks 为空，所以 exit=0 只证明 layer-0 控制流和 artifact 写出。",
            "",
            "该 smoke 使用本地兼容补丁，既不是原始 official commit 的成功，也不是完整模型 PPL/任务准确率、真实部署 payload 或同率方法比较。",
            "",
        ]
    lines = [
        "# LiftQuant 官方仓库兼容性审计",
        "",
        f"> 核验日期：{audit['verified_as_of']}；固定 commit：`{provenance['commit']}`。",
        "> 本审计程序没有启动训练或量化；下文另行汇总主任务已经结束的 bounded smoke 证据。没有产生可用于方法排名的精度结果。",
        "",
        "## 结论",
        "",
        "当前固定 commit **尚不能进入外部复现实验排名**。Block correction 与 optional E2E 必须分开看：",
        "",
        f"- Block correction：`{audit['compatibility']['block_correction']}`。`main.py --help` 返回码为 "
        f"`{main['help_probe']['returncode']}`，但 README 的 `{', '.join(main['unsupported_readme_flags'])}` 不存在于实际 parser。",
        f"- Optional E2E：`{audit['compatibility']['optional_e2e']}`。`e2efinetune.py --help` 返回码为 "
        f"`{e2e['help_probe']['returncode']}`，在解析参数前即失败。",
        f"- Compatibility-patched block smoke：`{audit['compatibility']['patched_layer0_smoke']}`；"
        "它只覆盖 Qwen2.5-3B 的 layer 0，不改变完整外部复现仍为 pending 的结论。",
        "- 因此方法矩阵保持 `external_reproduction_pending`，同时分别记录 official code audit、compatibility-patched layer-0 smoke 与 E2E import blocker；不能填写 PPL、准确率、速度或优胜结论。",
        "",
        "## 固定版本与工作树",
        "",
        f"- 期望 commit：`{provenance['expected_commit']}`；匹配：`{provenance['commit_matches']}`。",
        f"- 整体工作树干净：`{provenance['worktree_clean']}`；源代码工作树干净：`{provenance['source_worktree_clean']}`。",
        provenance_note,
        "",
        *smoke_lines,
        "## CLI 与 README 漂移",
        "",
        f"`main.py` 静态提取到 {len(main['static_flags'])} 个 long flags，README block 命令含 {len(main['readme_flags'])} 个。未实现的是：",
        "",
        "```text",
        " ".join(main["unsupported_readme_flags"]),
        "```",
        "",
        "建议的机械映射只是候选，不是已验证的论文协议：",
        "",
        "- `--nsamples 4096` → `--nsamples1 4096 --nsamples2 4096`；",
        "- `--epochs 2` → `--epochs1 2 --epochs2 2`。",
        "",
        "代码把两组样本的最大值作为加载池；当每阶段样本数等于最大值时，又各减去 `1/32`。因此上述映射加载 4096 条，但每个启用阶段实际迭代 3968 条。必须分别记录“池大小”和“每阶段优化 token 数”，且需要上游确认 2 epochs 是每阶段 2 个，还是总计 2 个。",
        "",
        "E2E 使用 `HfArgumentParser(..., return_remaining_strings=True)`，却没有消费 `extra_args`。即使补齐缺失模块，未知/过期参数仍可能被静默忽略，应改成非空即失败。",
        "",
        "## 数据与模型路径",
        "",
    ]
    for record in audit["paths"]["hardcoded_redpajama"]:
        lines.append(
            f"- `{record['file']}:{record['line']}`：`{record['path']}`；210 上存在：`{record['exists_on_audit_host']}`。"
        )
    lines.extend(
        [
            "",
            "这些路径没有由 README 的 `--cache_dir` 控制；首次加载 RedPajama 时仍会命中作者机器的绝对路径。Block 与 E2E 两条路径均需参数化并固定数据集 revision/split。",
            "",
            f"Qwen 缩小 smoke 候选缓存存在：`{model.get('exists')}`；snapshot revision：`{model.get('revision')}`；"
            f"索引引用 shard 缺失：`{len(model.get('missing_referenced_shards', []))}`；"
            f"最低文件/header gate：`{model.get('minimal_files_present')}`。这不是完整 shard 哈希或模型加载证明。",
            "",
            "## 环境一致性",
            "",
            f"README 建议 Python `{req['readme_python_version']}`；审计环境 Python "
            f"`{(req['installed_probe'].get('metadata') or {}).get('python')}`。",
            "",
            "| 依赖 | requirements | 已安装 | 判定 |",
            "|---|---:|---:|---|",
        ]
    )
    for row in req["comparison"]:
        lines.append(
            f"| {row['name']} | {row['exact_pin'] or 'unpinned'} | {row['installed']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "仓库直接 import、但 `requirements.txt` 未声明：`"
            + "`, `".join(req["imported_but_not_declared"])
            + "`。",
            "",
            "E2E 还在仓库内 import `datautils_block`，但固定 commit 没有该模块；因此 E2E `--help` 已构成确定性失败，不需要启动 GPU 才能发现。",
            "",
            "## Findings",
            "",
            "| 严重度 | 范围 | 证据 | 影响 |",
            "|---|---|---|---|",
        ]
    )
    for finding in audit["findings"]:
        evidence = finding["evidence"].replace("|", "\\|").replace("\n", " ")
        impact = finding["impact"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {finding['severity']} | {finding['scope']} | {evidence} | {impact} |")
    lines.extend(
        [
            "",
            "## 后续命令模板（审计脚本未运行）",
            "",
            "### 更小的 Qwen2.5-3B 零优化 smoke",
            "",
            audit["commands"]["reduced_qwen_smoke"]["limitations"],
            "",
            "```bash",
            audit["commands"]["reduced_qwen_smoke"]["command"],
            "```",
            "",
            "### Block correction 映射后 full 模板",
            "",
            "该模板仍被 RedPajama 绝对路径和两阶段 epoch 语义阻塞，不能直接当作复现命令。",
            "",
            "```bash",
            audit["commands"]["block_correction_full_after_path_fix"]["command"],
            "```",
            "",
            "## 进入正式比较前的闸门",
            "",
            "1. 上游确认并修复 README 的两组 flag 映射，记录 resolved args。",
            "2. 参数化 RedPajama 路径，固定 revision、split、样本去重与实际 token 数。",
            "3. 补齐 E2E 缺失模块，并令未知 `extra_args` 直接报错。",
            "4. 用 Python 3.12 建立可满足的锁定环境，验证 Torch/CUDA/BitBLAS ABI；block 与 E2E 使用独立环境记录。",
            "5. 先运行缩小 smoke，再单独排队正式 block correction；E2E 属于微调 lane，不与 frozen/no-backward PTQ 混排。",
            "6. 正式结果必须记录 GPU-hours、峰值显存、训练/校准 tokens、随机种子、真实 artifact bytes、kernel 与相同 endpoint。",
            "",
            "完整机器可读证据见 `audit.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(audit: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    summary_path = output_dir / "summary.md"
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary_path.write_text(render_summary(audit), encoding="utf-8", newline="\n")
    return [json_path, summary_path]


def _parse_probe(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path probe must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("path probe must be NAME=PATH")
    return name.strip(), Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="pinned LiftQuant checkout")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="isolated Python executable")
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT)
    parser.add_argument("--model-cache", type=Path, default=None)
    parser.add_argument("--path-probe", action="append", type=_parse_probe, default=[])
    parser.add_argument("--patch-file", type=Path, default=None)
    parser.add_argument("--official-smoke-run", type=Path, default=None)
    parser.add_argument("--patched-smoke-run", type=Path, default=None)
    parser.add_argument("--patched-smoke-gpu", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout_seconds < 1 or args.timeout_seconds > 120:
        raise SystemExit("--timeout-seconds must be between 1 and 120")
    audit = build_audit(
        args.repo,
        python=args.python,
        expected_commit=args.expected_commit or None,
        model_cache=args.model_cache,
        path_probes=args.path_probe,
        patch_file=args.patch_file,
        official_smoke_run=args.official_smoke_run,
        patched_smoke_run=args.patched_smoke_run,
        patched_smoke_gpu=args.patched_smoke_gpu,
        timeout_seconds=args.timeout_seconds,
    )
    for path in write_outputs(audit, args.output_dir):
        print(f"wrote: {path}")


if __name__ == "__main__":
    main()
