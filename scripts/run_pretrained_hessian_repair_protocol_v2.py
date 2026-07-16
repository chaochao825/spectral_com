from __future__ import annotations

"""Protocol-aware v2 entry point for exact-rate Hessian-repair experiments.

The historical signed pilot remains bound to its recorded source snapshot.
This entry point reuses the current compression implementation while replacing
every text-based data path with token tensors reconstructed from the
preregistered confirmatory protocol.  Endpoint evaluation is routed to the
test partition, whereas radial comfort-path fitting is validation-only.  A
completion marker is published only after protocol provenance has been written
and checked.
"""

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# The protocol is cache-only by construction.  Set these before importing any
# Transformers modules so optional TensorFlow/Flax stacks and network probes
# cannot perturb or block the numerical run.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import torch

import confirmatory_protocol_windows as protocol
import run_pretrained_hessian_repair as legacy
import run_pretrained_llm_orthogonality as base


PROTOCOL_SPLIT_POLICY = (
    "preregistered_per_seed_disjoint_calibration_and_fixed_shared_test_windows"
)
FROZEN_HF_FILE_SHA256 = {
    "model.safetensors": "ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15",
    "tokenizer.json": "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624",
    "config.json": "002050231a9b1ec3ac77aa6b9b3bbdc4d923f4068a7dd33b8da72a9bd6ad9a43",
    "tokenizer_config.json": "70e38394e494931c6f773ba41e19460dd4436526b852207367f04341b4066d3f",
}


@dataclass
class _RuntimeState:
    args: argparse.Namespace | None = None
    bundle: protocol.ConfirmatoryProtocolSelection | None = None
    tokenizer: object | None = None
    deferred_output_dir: Path | None = None
    model_binding: dict[str, object] | None = None
    activation_sample_audit: dict[str, object] | None = None


_STATE = _RuntimeState()
_ORIGINAL_BUILD_ARG_PARSER = legacy.build_arg_parser
_ORIGINAL_LOAD_EVAL_TEXTS = base.load_eval_texts
_ORIGINAL_SPLIT_TEXTS = legacy.split_content_disjoint_text_windows
_ORIGINAL_TEXT_DIGEST = legacy._text_digest
_ORIGINAL_SOURCE_SNAPSHOT = legacy._source_snapshot
_ORIGINAL_MARK_COMPLETE = legacy.mark_output_complete
_ORIGINAL_WINDOW_EVALUATOR = legacy.evaluate_current_model_with_windows
_ORIGINAL_BASE_EVALUATOR = base.evaluate_current_model
_ORIGINAL_COVARIANCE_COLLECTOR = base.collect_activation_covariances
_ORIGINAL_SAMPLE_COLLECTOR = base.collect_activation_samples
_ORIGINAL_MODEL_LOADER = base.load_model_and_tokenizer_from_config


class ProtocolWindowSelection(Sequence[str]):
    """A lazy, hash-friendly view of one role in the consumed protocol."""

    def __init__(self, role: str, expected_count: int) -> None:
        self.role = str(role)
        self.expected_count = int(expected_count)

    def _windows(self, tokenizer: object | None = None) -> tuple[protocol.ProtocolWindow, ...]:
        bundle = _ensure_protocol(tokenizer)
        if self.role == "calibration":
            windows = bundle.calibration_windows
        elif self.role == "evaluation":
            windows = bundle.evaluation_windows
        elif self.role == "validation":
            windows = bundle.validation_windows
        elif self.role == "test":
            windows = bundle.test_windows
        else:  # pragma: no cover - constructor sites are fixed below
            raise RuntimeError(f"unsupported protocol selection role: {self.role}")
        if len(windows) != self.expected_count:
            raise RuntimeError(
                f"protocol {self.role} window count changed: {len(windows)} != {self.expected_count}"
            )
        return tuple(windows)

    def __len__(self) -> int:
        return self.expected_count

    def __getitem__(self, index: int | slice) -> str | list[str]:
        ids = [window.window_id for window in self._windows()]
        return ids[index]

    def __iter__(self) -> Iterator[str]:
        # The legacy run-config overlap check only needs stable identities.  The
        # numerical paths below call ``_windows`` directly and consume tokens.
        return iter(window.window_id for window in self._windows())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_header(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    path = Path(args.protocol_manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"protocol manifest does not exist: {path}")
    actual_sha256 = _sha256_file(path)
    expected_sha256 = str(args.protocol_manifest_sha256).lower()
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise RuntimeError(
            f"protocol manifest SHA-256 differs: {actual_sha256} != {expected_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("protocol manifest root must be an object")
    return payload, path


def _protocol_counts(args: argparse.Namespace) -> tuple[int, int, int]:
    payload, _path = _manifest_header(args)
    counts = payload.get("allocation_counts")
    if not isinstance(counts, dict):
        raise RuntimeError("protocol manifest has no allocation_counts object")
    calibration = int(counts.get("calibration_windows_per_seed", -1))
    role_key = "test_windows" if args.protocol_eval_role == "test" else "validation_windows"
    evaluation = int(counts.get(role_key, -1))
    validation = int(counts.get("validation_windows", -1))
    window_length = int((payload.get("tokenization") or {}).get("window_token_length", -1))
    if calibration != int(args.calib_limit):
        raise RuntimeError(
            f"--calib-limit must equal protocol windows: {args.calib_limit} != {calibration}"
        )
    if evaluation != int(args.eval_limit):
        raise RuntimeError(
            f"--eval-limit must equal protocol {args.protocol_eval_role} windows: "
            f"{args.eval_limit} != {evaluation}"
        )
    if window_length != int(args.sequence_length):
        raise RuntimeError(
            f"--sequence-length must equal protocol window length: "
            f"{args.sequence_length} != {window_length}"
        )
    return calibration, evaluation, validation


def _ensure_protocol(tokenizer: object | None = None) -> protocol.ConfirmatoryProtocolSelection:
    if _STATE.bundle is not None:
        if tokenizer is not None and _STATE.tokenizer is not tokenizer:
            raise RuntimeError("protocol was resolved with a different tokenizer object")
        return _STATE.bundle
    if _STATE.args is None:
        raise RuntimeError("protocol runtime arguments have not been initialized")
    if tokenizer is None:
        tokenizer = _STATE.tokenizer
    if tokenizer is None:
        raise RuntimeError("protocol token windows cannot be resolved before tokenizer loading")
    args = _STATE.args
    bundle = protocol.consume_confirmatory_protocol(
        args.protocol_manifest,
        expected_sha256=args.protocol_manifest_sha256,
        experiment_seed=int(args.protocol_seed),
        tokenizer=tokenizer,
        evaluation_role=args.protocol_eval_role,
    )
    if int(args.seed) != int(bundle.provenance.selected_seed):
        raise RuntimeError(
            f"compression seed differs from protocol seed: {args.seed} != "
            f"{bundle.provenance.selected_seed}"
        )
    _STATE.bundle = bundle
    _STATE.tokenizer = tokenizer
    return bundle


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = _ORIGINAL_BUILD_ARG_PARSER()
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--protocol-manifest-sha256", required=True)
    parser.add_argument("--protocol-seed", type=int, required=True)
    parser.add_argument(
        "--protocol-eval-role",
        choices=["test"],
        default="test",
    )
    return parser


def _load_protocol_placeholder_texts(
    args: argparse.Namespace,
    *,
    limit: int,
) -> tuple[list[str], str, list[dict[str, object]]]:
    del limit
    _STATE.args = args
    if str(args.model) != protocol.MODEL_ID:
        raise RuntimeError(
            f"protocol model differs from the CLI model: {args.model!r} != {protocol.MODEL_ID!r}"
        )
    if str(args.revision) != protocol.MODEL_SNAPSHOT_COMMIT:
        raise RuntimeError(
            "protocol model revision differs from the frozen snapshot: "
            f"{args.revision!r} != {protocol.MODEL_SNAPSHOT_COMMIT!r}"
        )
    if not bool(args.local_files_only):
        raise RuntimeError("protocol execution requires --local-files-only")
    calibration, evaluation, validation = _protocol_counts(args)
    total = calibration + evaluation + validation
    return (
        ["protocol-window-placeholder"] * total,
        # Preserve the legacy runner's real-WikiText fail-closed prefix while
        # making the manifest-bound source explicit.
        "dataset:wikitext:protocol_manifest",
        [
            {
                "source": "protocol_manifest",
                "manifest": str(args.protocol_manifest),
                "manifest_sha256": str(args.protocol_manifest_sha256),
                "protocol_seed": int(args.protocol_seed),
                "evaluation_role": str(args.protocol_eval_role),
                "window_count": total,
                "fallback_allowed": False,
            }
        ],
    )


def _load_protocol_bound_model_and_tokenizer(
    config: dict[str, object],
) -> tuple[torch.nn.Module, object, str]:
    if str(config.get("model")) != protocol.MODEL_ID:
        raise RuntimeError("model loader received an identity outside the frozen protocol")
    if str(config.get("revision")) != protocol.MODEL_SNAPSHOT_COMMIT:
        raise RuntimeError("model loader received a revision outside the frozen protocol")
    if config.get("local_files_only") is not True:
        raise RuntimeError("protocol model loading must remain local-cache-only")
    model, tokenizer, device = _ORIGINAL_MODEL_LOADER(config)
    identity = legacy._model_identity(model, tokenizer, protocol.MODEL_SNAPSHOT_COMMIT)
    required = {
        "requested_revision": protocol.MODEL_SNAPSHOT_COMMIT,
        "resolved_model_commit_hash": protocol.MODEL_SNAPSHOT_COMMIT,
    }
    for field, expected in required.items():
        if identity.get(field) != expected:
            raise RuntimeError(
                f"loaded protocol model identity {field} differs: "
                f"{identity.get(field)!r} != {expected!r}"
            )
    tokenizer_runtime_commit = identity.get("resolved_tokenizer_commit_hash")
    if tokenizer_runtime_commit not in {None, protocol.MODEL_SNAPSHOT_COMMIT}:
        raise RuntimeError(
            "loaded tokenizer exposed a conflicting snapshot commit: "
            f"{tokenizer_runtime_commit!r}"
        )
    if identity.get("model_name_or_path") != protocol.MODEL_ID:
        raise RuntimeError("loaded model name_or_path differs from the frozen protocol model")
    if identity.get("tokenizer_name_or_path") != protocol.MODEL_ID:
        raise RuntimeError("loaded tokenizer name_or_path differs from the frozen protocol model")
    if model.__class__.__name__ != "GPTNeoXForCausalLM":
        raise RuntimeError(
            f"loaded protocol model class differs: {model.__class__.__name__!r}"
        )
    from transformers.utils.hub import cached_file

    snapshot_files: list[dict[str, object]] = []
    for filename, expected_sha256 in FROZEN_HF_FILE_SHA256.items():
        resolved = cached_file(
            protocol.MODEL_ID,
            filename,
            revision=protocol.MODEL_SNAPSHOT_COMMIT,
            local_files_only=True,
        )
        if resolved is None:
            raise RuntimeError(f"frozen protocol asset is unavailable: {filename}")
        asset_path = Path(resolved)
        normalized = asset_path.as_posix()
        snapshot_fragment = f"/snapshots/{protocol.MODEL_SNAPSHOT_COMMIT}/"
        if snapshot_fragment not in normalized:
            raise RuntimeError(
                f"frozen protocol asset did not resolve through the pinned snapshot: {resolved}"
            )
        actual_sha256 = _sha256_file(asset_path)
        if not hmac.compare_digest(actual_sha256, expected_sha256):
            raise RuntimeError(
                f"frozen protocol asset SHA differs for {filename}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        snapshot_files.append(
            {
                "filename": filename,
                "sha256": actual_sha256,
                "size_bytes": asset_path.stat().st_size,
                "snapshot_commit": protocol.MODEL_SNAPSHOT_COMMIT,
            }
        )
    _STATE.model_binding = {
        **identity,
        "expected_model_id": protocol.MODEL_ID,
        "expected_snapshot_commit": protocol.MODEL_SNAPSHOT_COMMIT,
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_runtime_commit_attestation": (
            "exact" if tokenizer_runtime_commit is not None else "runtime_field_unavailable_asset_sha_bound"
        ),
        "snapshot_files": snapshot_files,
        "validated": True,
    }
    return model, tokenizer, device


def _split_protocol_windows(args: argparse.Namespace, texts: list[str]) -> None:
    del texts
    _STATE.args = args
    calibration, evaluation, validation = _protocol_counts(args)
    args.calib_texts = ProtocolWindowSelection("calibration", calibration)
    args.eval_texts = ProtocolWindowSelection("evaluation", evaluation)
    args.recovery_texts = ProtocolWindowSelection("validation", validation)
    args.comfort_texts = args.recovery_texts
    args.comfort_eval_limit = validation
    args.comfort_evidence_role = "protocol_validation_only"
    args.text_split_policy = PROTOCOL_SPLIT_POLICY
    args.unique_text_pool_count = calibration + evaluation + validation


def _protocol_batches(
    selection: ProtocolWindowSelection,
    tokenizer: object,
    *,
    sequence_length: int,
    batch_size: int,
    limit: int,
) -> Iterator[tuple[list[protocol.ProtocolWindow], torch.Tensor]]:
    windows = selection._windows(tokenizer)
    if sequence_length != protocol.WINDOW_TOKEN_LENGTH:
        raise RuntimeError(
            f"protocol batches require {protocol.WINDOW_TOKEN_LENGTH} tokens, got {sequence_length}"
        )
    if limit != len(windows):
        raise RuntimeError(f"protocol limit must consume every selected window: {limit} != {len(windows)}")
    step = max(int(batch_size), 1)
    for start in range(0, len(windows), step):
        chosen = list(windows[start : start + step])
        tensor = torch.tensor([window.token_ids for window in chosen], dtype=torch.long)
        if tensor.shape != (len(chosen), sequence_length):
            raise RuntimeError(f"invalid reconstructed protocol tensor shape: {tuple(tensor.shape)}")
        yield chosen, tensor


def evaluate_current_model_with_protocol_windows(
    model: torch.nn.Module,
    tokenizer: object,
    *,
    strategy: str,
    texts: ProtocolWindowSelection,
    sequence_length: int,
    batch_size: int,
    device: str,
    eval_limit: int,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    if not isinstance(texts, ProtocolWindowSelection):
        raise TypeError("protocol v2 evaluator refuses non-protocol text input")
    nll_total = 0.0
    token_total = 0
    rows: list[dict[str, object]] = []
    window_index = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (windows, batch) in enumerate(
            _protocol_batches(
                texts,
                tokenizer,
                sequence_length=sequence_length,
                batch_size=batch_size,
                limit=eval_limit,
            )
        ):
            batch = batch.to(device)
            outputs = model(input_ids=batch)
            logits = outputs.logits[:, :-1, :].float()
            labels = batch[:, 1:]
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            for sequence_index, window in enumerate(windows):
                window_sum = float(losses[sequence_index].sum().detach().cpu())
                window_tokens = int(losses.shape[1])
                window_nll = window_sum / max(window_tokens, 1)
                rows.append(
                    {
                        "strategy": strategy,
                        "window_index": window_index,
                        "batch_index": batch_index,
                        "sequence_index": sequence_index,
                        "tokens": window_tokens,
                        "nll_sum": window_sum,
                        "nll": window_nll,
                        "perplexity": float(math.exp(min(window_nll, 50.0))),
                        "protocol_window_id": window.window_id,
                        "protocol_window_sha256": window.token_digest,
                        "protocol_role": window.role,
                        "protocol_seed": window.seed,
                    }
                )
                window_index += 1
                nll_total += window_sum
                token_total += window_tokens
    mean_nll = nll_total / max(token_total, 1)
    return (
        {
            "nll": mean_nll,
            "perplexity": float(math.exp(min(mean_nll, 50.0))),
            "tokens": token_total,
        },
        rows,
    )


def _evaluate_current_protocol_model(
    model: torch.nn.Module,
    tokenizer: object,
    *,
    texts: ProtocolWindowSelection,
    sequence_length: int,
    batch_size: int,
    device: str,
    eval_limit: int,
) -> dict[str, float | int]:
    metrics, _rows = evaluate_current_model_with_protocol_windows(
        model,
        tokenizer,
        strategy="internal",
        texts=texts,
        sequence_length=sequence_length,
        batch_size=batch_size,
        device=device,
        eval_limit=eval_limit,
    )
    return metrics


def collect_protocol_activation_covariances(
    model: torch.nn.Module,
    tokenizer: object,
    modules: dict[str, torch.nn.Linear],
    *,
    texts: ProtocolWindowSelection,
    sequence_length: int,
    batch_size: int,
    device: str,
    calib_limit: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    if not isinstance(texts, ProtocolWindowSelection):
        raise TypeError("protocol covariance collector refuses non-protocol text input")
    accum = {
        name: torch.zeros(module.weight.shape[1], module.weight.shape[1], dtype=torch.float64)
        for name, module in modules.items()
    }
    counts = {name: 0 for name in modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            value = inputs[0].detach()
            flat = value.reshape(1, -1) if value.ndim == 1 else value.reshape(-1, value.shape[-1])
            flat = flat.float()
            accum[name] += flat.transpose(0, 1).matmul(flat).double().cpu()
            counts[name] += int(flat.shape[0])

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        model.eval()
        with torch.no_grad():
            for _windows, batch in _protocol_batches(
                texts,
                tokenizer,
                sequence_length=sequence_length,
                batch_size=batch_size,
                limit=calib_limit,
            ):
                model(input_ids=batch.to(device))
    finally:
        for handle in handles:
            handle.remove()

    covariances: dict[str, torch.Tensor] = {}
    for name, covariance in accum.items():
        covariance = covariance / float(max(counts[name], 1))
        diagonal_mean = float(torch.diag(covariance).mean().item()) if covariance.numel() else 1.0
        ridge = max(diagonal_mean, base.EPS) * 1e-5
        covariances[name] = (
            covariance + torch.eye(covariance.shape[0], dtype=torch.float64) * ridge
        ).float()
    return covariances, counts


def collect_protocol_activation_samples(
    model: torch.nn.Module,
    tokenizer: object,
    modules: dict[str, torch.nn.Linear],
    *,
    texts: ProtocolWindowSelection,
    sequence_length: int,
    batch_size: int,
    device: str,
    calib_limit: int,
    max_rows: int,
) -> dict[str, torch.Tensor]:
    if not isinstance(texts, ProtocolWindowSelection):
        raise TypeError("protocol activation sampler refuses non-protocol text input")
    if max_rows <= 0:
        return {
            name: torch.empty(0, module.weight.shape[1], dtype=torch.float32)
            for name, module in modules.items()
        }
    windows = texts._windows(tokenizer)
    total_rows = len(windows) * int(sequence_length)
    sample_count = min(int(max_rows), total_rows)
    if sample_count <= 0:
        return {
            name: torch.empty(0, module.weight.shape[1], dtype=torch.float32)
            for name, module in modules.items()
        }
    if sample_count == 1:
        target_rows = torch.tensor([0], dtype=torch.long)
    else:
        target_rows = torch.linspace(0, total_rows - 1, steps=sample_count).round().long()
    if int(torch.unique(target_rows).numel()) != sample_count:
        raise RuntimeError("deterministic protocol sample row selection produced duplicates")
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
    counts = {name: 0 for name in modules}
    seen_rows = {name: 0 for name in modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            value = inputs[0].detach()
            flat = value.reshape(1, -1) if value.ndim == 1 else value.reshape(-1, value.shape[-1])
            start = seen_rows[name]
            stop = start + int(flat.shape[0])
            mask = (target_rows >= start) & (target_rows < stop)
            local_rows = target_rows[mask] - start
            if int(local_rows.numel()) > 0:
                chunks[name].append(flat[local_rows.to(flat.device)].float().cpu())
                counts[name] += int(local_rows.numel())
            seen_rows[name] = stop

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        model.eval()
        with torch.no_grad():
            for _windows, batch in _protocol_batches(
                texts,
                tokenizer,
                sequence_length=sequence_length,
                batch_size=batch_size,
                limit=calib_limit,
            ):
                model(input_ids=batch.to(device))
    finally:
        for handle in handles:
            handle.remove()

    if any(seen != total_rows for seen in seen_rows.values()):
        raise RuntimeError(
            f"activation sampler did not traverse every calibration token row: {seen_rows}"
        )
    if any(count != sample_count for count in counts.values()):
        raise RuntimeError(
            f"activation sampler did not retain the frozen sample count: {counts}"
        )
    _STATE.activation_sample_audit = {
        "policy": "deterministic_evenly_spaced_over_all_calibration_token_rows",
        "calibration_window_ids": list(_ensure_protocol().provenance.calibration_window_ids),
        "calibration_window_count": len(windows),
        "total_token_rows": total_rows,
        "sampled_rows_per_selected_tensor": sample_count,
        "all_calibration_windows_traversed": True,
    }

    return {
        name: torch.cat(chunks[name], dim=0)
        if chunks[name]
        else torch.empty(0, module.weight.shape[1], dtype=torch.float32)
        for name, module in modules.items()
    }


def _protocol_text_digest(texts: Sequence[object]) -> str:
    if not isinstance(texts, ProtocolWindowSelection):
        return _ORIGINAL_TEXT_DIGEST(texts)
    bundle = _ensure_protocol()
    provenance = bundle.provenance
    if texts.role == "calibration":
        return provenance.calibration_token_sha256
    if texts.role == "evaluation":
        return provenance.evaluation_token_sha256
    if texts.role == "validation":
        return provenance.validation_token_sha256
    if texts.role == "test":
        return provenance.test_token_sha256
    raise RuntimeError(f"unknown protocol digest role: {texts.role}")


def _protocol_source_snapshot() -> dict[str, dict[str, object]]:
    if _STATE.args is None:
        raise RuntimeError("protocol source snapshot requested before argument initialization")
    repo_root = Path(__file__).resolve().parents[1]
    manifest = Path(_STATE.args.protocol_manifest).expanduser().resolve()
    candidates = {
        "runner": Path(__file__).resolve(),
        "protocol_consumer": Path(protocol.__file__).resolve(),
        "legacy_runner": Path(legacy.__file__).resolve(),
        "codec": repo_root / "src" / "llm_spectral_dynamics" / "structured" / "codec_artifact.py",
        "hessian_repair": repo_root / "src" / "llm_spectral_dynamics" / "structured" / "hessian_repair.py",
        "base_runner": Path(base.__file__).resolve(),
        "model_data": repo_root / "src" / "llm_spectral_dynamics" / "structured" / "data.py",
        "protocol_manifest": manifest,
    }
    snapshot: dict[str, dict[str, object]] = {}
    for name, path in candidates.items():
        raw = path.read_bytes()
        try:
            rendered = path.relative_to(repo_root).as_posix()
        except ValueError:
            rendered = str(path)
        snapshot[name] = {
            "path": rendered,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if snapshot["protocol_manifest"]["sha256"] != str(
        _STATE.args.protocol_manifest_sha256
    ).lower():
        raise RuntimeError("source snapshot protocol SHA differs from the CLI contract")
    return snapshot


def _defer_completion(path: Path) -> None:
    target = Path(path)
    if _STATE.deferred_output_dir is not None and _STATE.deferred_output_dir != target:
        raise RuntimeError("multiple output directories reached the protocol completion gate")
    _STATE.deferred_output_dir = target


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.protocol-v2.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _validate_protocol_window_csv(
    path: Path,
    bundle: protocol.ConfirmatoryProtocolSelection,
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_windows = list(bundle.evaluation_windows)
    expected_ids = [window.window_id for window in expected_windows]
    expected_hashes = [window.token_digest for window in expected_windows]
    expected_roles = [window.role for window in expected_windows]
    expected_seeds = ["" if window.seed is None else str(window.seed) for window in expected_windows]
    strategies: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        strategies.setdefault(str(row.get("strategy", "")), []).append(row)
    expected_strategies = ["dense", *legacy.STRATEGY_ORDER]
    if list(strategies) != expected_strategies:
        raise RuntimeError(
            f"endpoint window strategies differ from protocol contract: "
            f"{list(strategies)} != {expected_strategies}"
        )
    for strategy, group in strategies.items():
        observed_ids = [row.get("protocol_window_id") for row in group]
        observed_hashes = [row.get("protocol_window_sha256") for row in group]
        if observed_ids != expected_ids or observed_hashes != expected_hashes:
            raise RuntimeError(f"{strategy} did not consume the ordered protocol windows")
        if [row.get("protocol_role") for row in group] != expected_roles:
            raise RuntimeError(f"{strategy} protocol window roles differ from the manifest")
        if [row.get("protocol_seed") for row in group] != expected_seeds:
            raise RuntimeError(f"{strategy} protocol window seeds differ from the manifest")
        if [int(row.get("tokens", -1)) for row in group] != [
            protocol.WINDOW_TOKEN_LENGTH - 1
        ] * len(expected_windows):
            raise RuntimeError(f"{strategy} protocol token counts differ from the manifest")
        indices = [int(row.get("window_index", -1)) for row in group]
        if indices != list(range(len(expected_windows))):
            raise RuntimeError(f"{strategy} protocol window indices are not contiguous")


def _augment_protocol_outputs(output_dir: Path) -> None:
    if _STATE.args is None or _STATE.bundle is None:
        raise RuntimeError("protocol output augmentation requires a consumed protocol bundle")
    if _STATE.model_binding is None:
        raise RuntimeError("protocol output augmentation requires a validated model binding")
    if _STATE.activation_sample_audit is None:
        raise RuntimeError("protocol output augmentation requires a complete activation-sample audit")
    run_config_path = output_dir / "run_config.json"
    payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise RuntimeError("legacy runner produced no auditable data block")
    provenance = asdict(_STATE.bundle.provenance)
    # Keep the committed locator portable while the SHA binds its bytes.
    provenance["manifest_path"] = str(_STATE.args.protocol_manifest)
    data = payload["data"]
    reconstructed_windows = {
        window.window_id: window
        for window in (
            *_STATE.bundle.calibration_windows,
            *_STATE.bundle.validation_windows,
            *_STATE.bundle.test_windows,
        )
    }
    selected_window_count = len(reconstructed_windows)
    data.update(
        {
            "requested": {
                "dataset": protocol.DATASET_NAME,
                "subset": protocol.DATASET_CONFIG,
                "calibration_split": "train",
                "evaluation_split": str(_STATE.args.protocol_eval_role),
                "sequence_length": protocol.WINDOW_TOKEN_LENGTH,
                "batch_size": int(_STATE.args.batch_size),
                "allow_fallback": False,
            },
            "source_used": "dataset:wikitext:protocol_manifest",
            "fallback_allowed": False,
            "split_policy": PROTOCOL_SPLIT_POLICY,
            "text_pool_count": selected_window_count,
            "unique_text_pool_count": selected_window_count,
            "text_pool_count_semantics": (
                "unique reconstructed/available protocol token windows for the selected seed; "
                "this is not a claim that every available window enters every numerical path"
            ),
            "calib_text_count": _STATE.bundle.provenance.calibration_window_count,
            "eval_text_count": _STATE.bundle.provenance.evaluation_window_count,
            "eval_window_count": _STATE.bundle.provenance.evaluation_window_count,
            "window_interval_semantics": (
                "fixed shared test windows support paired diagnostics; the seed-level "
                "replicate is the disjoint calibration allocation, not each window"
            ),
            "calib_digest": _STATE.bundle.provenance.calibration_token_sha256,
            "eval_digest": _STATE.bundle.provenance.evaluation_token_sha256,
            "identical_text_overlap_count": 0,
            "content_disjoint": True,
            "protocol": provenance,
            "protocol_activation_sampling": _STATE.activation_sample_audit,
            "protocol_numerical_path_window_counts": {
                "covariance_calibration": _STATE.bundle.provenance.calibration_window_count,
                "activation_risk_calibration": _STATE.bundle.provenance.calibration_window_count,
                "endpoint_nll_evaluation": _STATE.bundle.provenance.evaluation_window_count,
                "comfort_recovery_validation": (
                    0 if bool(_STATE.args.skip_comfort) else len(_STATE.bundle.validation_windows)
                ),
                "reconstructed_available_unique": selected_window_count,
            },
        }
    )
    payload["protocol_model_binding"] = _STATE.model_binding
    payload["protocol_consumer"] = {
        "version": protocol.SCHEMA_VERSION,
        "direct_token_tensor_input": True,
        "text_join_or_retokenization": False,
        "token_repetition": False,
    }
    _validate_protocol_window_csv(output_dir / "endpoint_window_nll.csv", _STATE.bundle)
    if not bool(_STATE.args.skip_comfort):
        comfort_path = output_dir / "comfort_sweep.csv"
        with comfort_path.open("r", encoding="utf-8", newline="") as handle:
            comfort_rows = list(csv.DictReader(handle))
        if not comfort_rows or {
            str(row.get("evidence_role", "")) for row in comfort_rows
        } != {"protocol_validation_only"}:
            raise RuntimeError("comfort sweep did not remain on protocol validation windows")
    _write_json_atomic(run_config_path, payload)


def _install_protocol_patches() -> None:
    legacy.build_arg_parser = _build_arg_parser
    base.load_eval_texts = _load_protocol_placeholder_texts
    legacy.split_content_disjoint_text_windows = _split_protocol_windows
    legacy._text_digest = _protocol_text_digest
    legacy._source_snapshot = _protocol_source_snapshot
    legacy.mark_output_complete = _defer_completion
    legacy.evaluate_current_model_with_windows = evaluate_current_model_with_protocol_windows
    base.evaluate_current_model = _evaluate_current_protocol_model
    base.collect_activation_covariances = collect_protocol_activation_covariances
    base.collect_activation_samples = collect_protocol_activation_samples
    base.load_model_and_tokenizer_from_config = _load_protocol_bound_model_and_tokenizer


def main() -> None:
    _install_protocol_patches()
    legacy.main()
    if _STATE.deferred_output_dir is None:
        raise RuntimeError("legacy runner did not reach the protocol completion gate")
    _augment_protocol_outputs(_STATE.deferred_output_dir)
    _ORIGINAL_MARK_COMPLETE(_STATE.deferred_output_dir)


if __name__ == "__main__":
    main()
