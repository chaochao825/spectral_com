from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_pretrained_hessian_repair_protocol_v2",
    SCRIPTS / "run_pretrained_hessian_repair_protocol_v2.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _window(window_id: str, role: str, seed: int | None, offset: int) -> object:
    tokens = tuple((offset + index) % 11 for index in range(RUNNER.protocol.WINDOW_TOKEN_LENGTH))
    return RUNNER.protocol.ProtocolWindow(
        window_id=window_id,
        role=role,
        seed=seed,
        token_ids=tokens,
        token_digest=RUNNER.protocol.token_ids_sha256(tokens),
    )


def _bundle() -> object:
    calibration = (
        _window("calibration/seed-17/000", "calibration", 17, 0),
        _window("calibration/seed-17/001", "calibration", 17, 1),
    )
    validation = (
        _window("validation/fixed/000", "validation", None, 2),
        _window("validation/fixed/001", "validation", None, 3),
    )
    test = (
        _window("test/fixed/000", "test", None, 4),
        _window("test/fixed/001", "test", None, 5),
    )
    provenance = RUNNER.protocol.ProtocolProvenance(
        manifest_path="results/protocol.json",
        manifest_sha256="a" * 64,
        schema_version=RUNNER.protocol.SCHEMA_VERSION,
        selected_seed=17,
        evaluation_role="test",
        window_token_length=RUNNER.protocol.WINDOW_TOKEN_LENGTH,
        calibration_window_ids=tuple(window.window_id for window in calibration),
        calibration_window_count=len(calibration),
        calibration_token_sha256=RUNNER.protocol.digest_protocol_windows(calibration),
        evaluation_window_ids=tuple(window.window_id for window in test),
        evaluation_window_count=len(test),
        evaluation_token_sha256=RUNNER.protocol.digest_protocol_windows(test),
        consumed=True,
        status=RUNNER.protocol.PROTOCOL_STATUS,
        protocol_date=RUNNER.protocol.PROTOCOL_DATE,
        model_id=RUNNER.protocol.MODEL_ID,
        model_snapshot_commit=RUNNER.protocol.MODEL_SNAPSHOT_COMMIT,
        tokenizer_class=RUNNER.protocol.TOKENIZER_CLASS,
        tokenizer_vocab_size=RUNNER.protocol.TOKENIZER_VOCAB_SIZE,
        tokenizer_snapshot_commit=RUNNER.protocol.MODEL_SNAPSHOT_COMMIT,
        dataset_name=RUNNER.protocol.DATASET_NAME,
        dataset_config=RUNNER.protocol.DATASET_CONFIG,
        dataset_fingerprints=RUNNER.protocol.DATASET_FINGERPRINTS,
        dataset_local_cache_only=True,
        dataset_fallback_allowed=False,
        tokenizer_add_special_tokens=False,
        tokenization_input="native raw dataset text without normalization",
        source_row_reuse_allowed=False,
        manifest_seeds=(17,),
        epsilon_grid=(0.0, 0.125, 1.0),
        local_fit_positive_epsilons=(0.125,),
        validation_window_ids=tuple(window.window_id for window in validation),
        validation_token_sha256=RUNNER.protocol.digest_protocol_windows(validation),
        test_window_ids=tuple(window.window_id for window in test),
        test_token_sha256=RUNNER.protocol.digest_protocol_windows(test),
        all_calibration_window_ids=tuple(window.window_id for window in calibration),
        calibration_token_sha256_by_seed=(
            (17, RUNNER.protocol.digest_protocol_windows(calibration)),
        ),
        all_window_count=6,
        all_window_token_sha256=RUNNER.protocol.digest_protocol_windows(
            (*test, *validation, *calibration)
        ),
        allocated_source_row_count=6,
        allocated_source_row_ids_sha256="b" * 64,
    )
    return RUNNER.protocol.ConfirmatoryProtocolSelection(
        selected_calibration_windows=calibration,
        evaluation_windows=test,
        validation_windows=validation,
        test_windows=test,
        provenance=provenance,
    )


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    RUNNER._STATE.args = None
    RUNNER._STATE.bundle = None
    RUNNER._STATE.tokenizer = None
    RUNNER._STATE.deferred_output_dir = None
    RUNNER._STATE.model_binding = None
    RUNNER._STATE.activation_sample_audit = None
    yield
    RUNNER._STATE.args = None
    RUNNER._STATE.bundle = None
    RUNNER._STATE.tokenizer = None
    RUNNER._STATE.deferred_output_dir = None
    RUNNER._STATE.model_binding = None
    RUNNER._STATE.activation_sample_audit = None


def test_protocol_batches_are_exact_tensor_windows_without_retokenization() -> None:
    tokenizer = object()
    RUNNER._STATE.bundle = _bundle()
    RUNNER._STATE.tokenizer = tokenizer
    selection = RUNNER.ProtocolWindowSelection("evaluation", 2)
    batches = list(
        RUNNER._protocol_batches(
            selection,
            tokenizer,
            sequence_length=256,
            batch_size=2,
            limit=2,
        )
    )
    assert len(batches) == 1
    windows, tensor = batches[0]
    assert tensor.dtype == torch.long
    assert tensor.tolist() == [list(window.token_ids) for window in windows]
    with pytest.raises(RuntimeError, match="consume every selected window"):
        list(
            RUNNER._protocol_batches(
                selection,
                tokenizer,
                sequence_length=256,
                batch_size=1,
                limit=1,
            )
        )


class TinyProtocolModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)

    def forward(self, input_ids: torch.Tensor) -> object:
        features = torch.stack(
            [input_ids.float() / 10.0, torch.ones_like(input_ids, dtype=torch.float32)],
            dim=-1,
        )
        self.proj(features)
        vocab = 11
        logits = torch.zeros(*input_ids.shape, vocab, dtype=torch.float32, device=input_ids.device)
        return SimpleNamespace(logits=logits)


def test_nll_and_activation_paths_consume_the_same_protocol_tensors() -> None:
    tokenizer = object()
    bundle = _bundle()
    RUNNER._STATE.bundle = bundle
    RUNNER._STATE.tokenizer = tokenizer
    model = TinyProtocolModel()
    evaluation = RUNNER.ProtocolWindowSelection("evaluation", 2)
    metrics, rows = RUNNER.evaluate_current_model_with_protocol_windows(
        model,
        tokenizer,
        strategy="dense",
        texts=evaluation,
        sequence_length=256,
        batch_size=1,
        device="cpu",
        eval_limit=2,
    )
    assert metrics["tokens"] == 2 * 255
    assert [row["protocol_window_id"] for row in rows] == list(
        bundle.provenance.evaluation_window_ids
    )
    assert [row["protocol_window_sha256"] for row in rows] == [
        window.token_digest for window in bundle.evaluation_windows
    ]

    calibration = RUNNER.ProtocolWindowSelection("calibration", 2)
    covariances, counts = RUNNER.collect_protocol_activation_covariances(
        model,
        tokenizer,
        {"proj": model.proj},
        texts=calibration,
        sequence_length=256,
        batch_size=1,
        device="cpu",
        calib_limit=2,
    )
    assert counts == {"proj": 512}
    assert covariances["proj"].shape == (2, 2)
    samples = RUNNER.collect_protocol_activation_samples(
        model,
        tokenizer,
        {"proj": model.proj},
        texts=calibration,
        sequence_length=256,
        batch_size=1,
        device="cpu",
        calib_limit=2,
        max_rows=9,
    )
    assert samples["proj"].shape == (9, 2)
    assert RUNNER._STATE.activation_sample_audit == {
        "policy": "deterministic_evenly_spaced_over_all_calibration_token_rows",
        "calibration_window_ids": list(bundle.provenance.calibration_window_ids),
        "calibration_window_count": 2,
        "total_token_rows": 512,
        "sampled_rows_per_selected_tensor": 9,
        "all_calibration_windows_traversed": True,
    }


def test_placeholder_source_passes_legacy_wikitext_gate_and_binds_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RUNNER, "_protocol_counts", lambda _args: (2, 2, 2))
    args = SimpleNamespace(
        model=RUNNER.protocol.MODEL_ID,
        revision=RUNNER.protocol.MODEL_SNAPSHOT_COMMIT,
        local_files_only=True,
        protocol_manifest="results/protocol.json",
        protocol_manifest_sha256="a" * 64,
        protocol_seed=17,
        protocol_eval_role="test",
    )
    texts, source, metadata = RUNNER._load_protocol_placeholder_texts(args, limit=999)
    assert source == "dataset:wikitext:protocol_manifest"
    assert source.startswith("dataset:wikitext")
    assert len(texts) == 6
    assert metadata[0]["fallback_allowed"] is False
    args.model = "EleutherAI/pythia-160m"
    with pytest.raises(RuntimeError, match="protocol model differs"):
        RUNNER._load_protocol_placeholder_texts(args, limit=999)


def _write_window_csv(path: Path, bundle: object, *, tamper: bool = False) -> None:
    fields = [
        "strategy",
        "window_index",
        "tokens",
        "protocol_window_id",
        "protocol_window_sha256",
        "protocol_role",
        "protocol_seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for strategy in ("dense", *RUNNER.legacy.STRATEGY_ORDER):
            for index, window in enumerate(bundle.evaluation_windows):
                writer.writerow(
                    {
                        "strategy": strategy,
                        "window_index": index,
                        "tokens": RUNNER.protocol.WINDOW_TOKEN_LENGTH - 1,
                        "protocol_window_id": (
                            "tampered" if tamper and strategy == "Q" and index == 0 else window.window_id
                        ),
                        "protocol_window_sha256": window.token_digest,
                        "protocol_role": window.role,
                        "protocol_seed": "" if window.seed is None else window.seed,
                    }
                )


def test_window_csv_gate_requires_identical_order_for_every_strategy(tmp_path: Path) -> None:
    bundle = _bundle()
    path = tmp_path / "endpoint_window_nll.csv"
    _write_window_csv(path, bundle)
    RUNNER._validate_protocol_window_csv(path, bundle)
    _write_window_csv(path, bundle, tamper=True)
    with pytest.raises(RuntimeError, match="ordered protocol windows"):
        RUNNER._validate_protocol_window_csv(path, bundle)


def test_protocol_augmentation_precedes_completion_and_is_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle()
    RUNNER._STATE.bundle = bundle
    RUNNER._STATE.tokenizer = object()
    RUNNER._STATE.args = SimpleNamespace(
        protocol_manifest="results/protocol.json",
        protocol_manifest_sha256="a" * 64,
        protocol_seed=17,
        protocol_eval_role="test",
        batch_size=1,
        skip_comfort=True,
    )
    RUNNER._STATE.model_binding = {"validated": True}
    RUNNER._STATE.activation_sample_audit = {
        "policy": "deterministic_evenly_spaced_over_all_calibration_token_rows",
        "calibration_window_ids": list(bundle.provenance.calibration_window_ids),
        "calibration_window_count": 2,
        "total_token_rows": 512,
        "sampled_rows_per_selected_tensor": 9,
        "all_calibration_windows_traversed": True,
    }
    (tmp_path / "run_config.json").write_text(
        json.dumps({"data": {"source_metadata": []}}), encoding="utf-8"
    )
    _write_window_csv(tmp_path / "endpoint_window_nll.csv", bundle)
    RUNNER._augment_protocol_outputs(tmp_path)
    payload = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert payload["data"]["protocol"]["consumed"] is True
    assert payload["data"]["protocol"]["evaluation_window_ids"] == list(
        bundle.provenance.evaluation_window_ids
    )
    assert payload["data"]["unique_text_pool_count"] == 6
    assert "protocol token windows" in payload["data"]["text_pool_count_semantics"]
    assert payload["protocol_consumer"] == {
        "version": RUNNER.protocol.SCHEMA_VERSION,
        "direct_token_tensor_input": True,
        "text_join_or_retokenization": False,
        "token_repetition": False,
    }
    assert not (tmp_path / "COMPLETED").exists()


def test_model_loader_fails_closed_on_protocol_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "model": RUNNER.protocol.MODEL_ID,
        "revision": RUNNER.protocol.MODEL_SNAPSHOT_COMMIT,
        "local_files_only": True,
    }

    class _Config:
        _commit_hash = RUNNER.protocol.MODEL_SNAPSHOT_COMMIT
        _name_or_path = RUNNER.protocol.MODEL_ID

        def to_dict(self) -> dict[str, object]:
            return {"model_type": "gpt_neox"}

    model_type = type("GPTNeoXForCausalLM", (), {"config": _Config()})
    tokenizer_type = type(
        "GPTNeoXTokenizerFast",
        (),
        {
            "init_kwargs": {"_commit_hash": None},
            "name_or_path": RUNNER.protocol.MODEL_ID,
        },
    )
    snapshot = tmp_path / "snapshots" / RUNNER.protocol.MODEL_SNAPSHOT_COMMIT
    snapshot.mkdir(parents=True)
    assets = {}
    for filename in ("model.safetensors", "tokenizer.json", "config.json", "tokenizer_config.json"):
        path = snapshot / filename
        path.write_bytes(f"frozen-{filename}".encode("utf-8"))
        assets[filename] = RUNNER._sha256_file(path)
    monkeypatch.setattr(RUNNER, "FROZEN_HF_FILE_SHA256", assets)
    import transformers.utils.hub

    monkeypatch.setattr(
        transformers.utils.hub,
        "cached_file",
        lambda _model, filename, **_kwargs: str(snapshot / filename),
    )
    monkeypatch.setattr(
        RUNNER,
        "_ORIGINAL_MODEL_LOADER",
        lambda _config: (model_type(), tokenizer_type(), "cpu"),
    )
    _model, _tokenizer, device = RUNNER._load_protocol_bound_model_and_tokenizer(config)
    assert device == "cpu"
    assert RUNNER._STATE.model_binding is not None
    assert RUNNER._STATE.model_binding["validated"] is True
    assert (
        RUNNER._STATE.model_binding["tokenizer_runtime_commit_attestation"]
        == "runtime_field_unavailable_asset_sha_bound"
    )
    with pytest.raises(RuntimeError, match="identity outside"):
        RUNNER._load_protocol_bound_model_and_tokenizer({**config, "model": "wrong"})

    bad_tokenizer_type = type(
        "GPTNeoXTokenizerFast",
        (),
        {
            "init_kwargs": {"_commit_hash": "f" * 40},
            "name_or_path": RUNNER.protocol.MODEL_ID,
        },
    )
    monkeypatch.setattr(
        RUNNER,
        "_ORIGINAL_MODEL_LOADER",
        lambda _config: (model_type(), bad_tokenizer_type(), "cpu"),
    )
    with pytest.raises(RuntimeError, match="conflicting snapshot commit"):
        RUNNER._load_protocol_bound_model_and_tokenizer(config)


def test_parser_exposes_protocol_binding_arguments() -> None:
    args = RUNNER._build_arg_parser().parse_args(
        [
            "--protocol-manifest",
            "protocol.json",
            "--protocol-manifest-sha256",
            "a" * 64,
            "--protocol-seed",
            "17",
            "--protocol-eval-role",
            "test",
        ]
    )
    assert args.protocol_seed == 17
    assert args.protocol_eval_role == "test"


def test_protocol_routes_comfort_only_to_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RUNNER, "_protocol_counts", lambda _args: (2, 3, 4))
    args = SimpleNamespace()
    RUNNER._split_protocol_windows(args, [])
    assert args.calib_texts.role == "calibration"
    assert args.eval_texts.role == "evaluation"
    assert args.eval_texts.expected_count == 3
    assert args.recovery_texts.role == "validation"
    assert args.comfort_texts is args.recovery_texts
    assert args.comfort_eval_limit == 4
    assert args.comfort_evidence_role == "protocol_validation_only"
