from pathlib import Path
from tempfile import TemporaryDirectory

from llm_spectral_dynamics.collect_activations import torch_dtype_from_name
from llm_spectral_dynamics import run_experiment
from llm_spectral_dynamics.run_experiment import _write_metric_delta_tables


class TorchStub:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"


def test_torch_dtype_from_name_handles_common_aliases():
    assert torch_dtype_from_name(TorchStub, None) is None
    assert torch_dtype_from_name(TorchStub, "auto") == "auto"
    assert torch_dtype_from_name(TorchStub, "bf16") == "bfloat16"
    assert torch_dtype_from_name(TorchStub, "fp16") == "float16"


def test_metric_delta_tables_include_pretrained_random_pairs():
    rows = [
        {
            "model": "m",
            "dataset_condition": "natural",
            "revision": "",
            "layer": 0,
            "site": "resid_post",
            "variant": "pretrained",
            "effective_rank": 10.0,
            "top_1_explained_variance": 0.4,
            "alpha": 1.5,
        },
        {
            "model": "m",
            "dataset_condition": "natural",
            "revision": "",
            "layer": 0,
            "site": "resid_post",
            "variant": "random",
            "effective_rank": 25.0,
            "top_1_explained_variance": 0.1,
            "alpha": 0.9,
        },
    ]
    with TemporaryDirectory() as tmp:
        paths = {"metrics": Path(tmp)}
        _write_metric_delta_tables(paths, rows)
        effective = (Path(tmp) / "effective_rank_delta.csv").read_text(encoding="utf-8")
        top1 = (Path(tmp) / "top1_delta.csv").read_text(encoding="utf-8")
        alpha = (Path(tmp) / "alpha_delta.csv").read_text(encoding="utf-8")
    assert "random_minus_pretrained" in effective
    assert "15.0" in effective
    assert "-0.30000000000000004" in top1
    assert "-0.6" in alpha


def test_metric_delta_tables_mark_pretrained_only_unavailable():
    rows = [
        {
            "model": "m",
            "dataset_condition": "natural",
            "revision": "",
            "layer": 0,
            "site": "resid_post",
            "variant": "pretrained",
            "effective_rank": 10.0,
            "top_1_explained_variance": 0.4,
            "alpha": 1.5,
        }
    ]
    with TemporaryDirectory() as tmp:
        paths = {"metrics": Path(tmp)}
        _write_metric_delta_tables(paths, rows)
        effective = (Path(tmp) / "effective_rank_delta.csv").read_text(encoding="utf-8")
        top1 = (Path(tmp) / "top1_delta.csv").read_text(encoding="utf-8")
        alpha = (Path(tmp) / "alpha_delta.csv").read_text(encoding="utf-8")
    assert "unavailable" in effective
    assert "requires matched pretrained and random variants" in effective
    assert "unavailable" in top1
    assert "unavailable" in alpha


def test_analysis_cli_overrides_nested_values():
    args = run_experiment.build_arg_parser().parse_args(
        [
            "--config",
            "configs/default.yaml",
            "--sample-limit",
            "512",
            "--bootstrap-samples",
            "16",
            "--powerlaw-rank-max",
            "64",
            "--dynamic-max-sequences",
            "4",
            "--dynamic-pca-rank",
            "16",
        ]
    )
    original = run_experiment.load_yaml
    run_experiment.load_yaml = lambda _path: {"analysis": {"dynamic": {}}}
    try:
        cfg = run_experiment._config_from_args(args)
    finally:
        run_experiment.load_yaml = original
    assert cfg["analysis"]["sample_limit"] == 512
    assert cfg["analysis"]["bootstrap_samples"] == 16
    assert cfg["analysis"]["powerlaw_rank_max"] == 64
    assert cfg["analysis"]["dynamic"]["max_sequences"] == 4
    assert cfg["analysis"]["dynamic"]["pca_rank"] == 16
