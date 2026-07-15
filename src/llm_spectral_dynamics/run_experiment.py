from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .collect_activations import collect_activation_statistics, collect_synthetic_statistics
from .config import deep_update, ensure_output_tree, load_yaml, parse_csv_arg, parse_layers
from .plots import plot_eigenspectrum, plot_eigenspectrum_overlay, plot_metric_heatmap
from .report import write_markdown_report


LOGGER = logging.getLogger("llm_spectral_dynamics")


def _jsonify(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonify(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _value_as_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _safe_name(text: str) -> str:
    return text.replace("/", "__").replace(":", "_").replace(" ", "_")


def _stage_existing_outputs_to_trash(output_dir: Path) -> Path | None:
    managed = [
        "metrics",
        "eigenvalues",
        "plots",
        "kv_cache",
        "interventions",
        "run_metadata.json",
        "report.md",
    ]
    existing = [output_dir / name for name in managed if (output_dir / name).exists()]
    if not existing:
        return None
    root = output_dir.resolve()
    trash_root = output_dir / "trash" / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-fresh-output"
    for src in existing:
        resolved = src.resolve()
        if not str(resolved).startswith(str(root)):
            raise RuntimeError(f"refusing to stage path outside output dir: {resolved}")
        rel = resolved.relative_to(root)
        dest = trash_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved), str(dest))
    return trash_root


def _save_pt(path: Path, payload: dict[str, object]) -> None:
    try:
        import torch

        torch.save(payload, path)
    except Exception:
        with path.open("wb") as handle:
            pickle.dump(payload, handle)


def _save_eigen_payloads(paths: dict[str, Path], payloads: list[dict[str, object]]) -> None:
    for payload in payloads:
        head_part = f"__head{payload['head']}" if "head" in payload else ""
        stem = _safe_name(
            f"{payload['model']}__{payload['variant']}__{payload['dataset_condition']}__layer{payload['layer']}{head_part}__{payload['site']}"
        )
        json_path = paths["eigenvalues"] / f"{stem}.json"
        pt_path = paths["eigenvalues"] / f"{stem}.pt"
        json_path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")
        _save_pt(pt_path, payload)


def _plot_outputs(paths: dict[str, Path], metric_rows: list[dict[str, object]], eigen_payloads: list[dict[str, object]]) -> None:
    for payload in eigen_payloads:
        head_part = f"__head{payload['head']}" if "head" in payload else ""
        stem = _safe_name(
            f"{payload['model']}__{payload['variant']}__layer{payload['layer']}{head_part}__{payload['site']}"
        )
        try:
            plot_eigenspectrum(
                np.asarray(payload["eigenvalues"], dtype=np.float64),
                paths["eigenspectra"] / f"{stem}.png",
                title=f"{payload['model']} {payload['variant']} L{payload['layer']} {payload['site']}",
            )
        except Exception as exc:
            LOGGER.warning("failed to plot eigenspectrum for %s: %s", stem, exc)
    for metric in ("alpha", "participation_ratio", "effective_rank"):
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in metric_rows:
            grouped.setdefault((str(row.get("model")), str(row.get("variant"))), []).append(row)
        for (model, variant), rows in grouped.items():
            stem = _safe_name(f"{model}__{variant}__{metric}")
            try:
                plot_metric_heatmap(rows, paths["heatmaps"] / f"{stem}.png", metric=metric, title=f"{model} {variant} {metric}")
            except Exception as exc:
                LOGGER.warning("failed to plot heatmap %s: %s", stem, exc)
    _plot_comparison_outputs(paths, eigen_payloads)


def _payload_group_key(payload: dict[str, object], *, omit_variant: bool) -> tuple[object, ...]:
    fields = ["model", "dataset_condition", "revision", "layer", "site", "head"]
    if not omit_variant:
        fields.insert(1, "variant")
    return tuple(payload.get(field, "") for field in fields)


def _plot_comparison_outputs(paths: dict[str, Path], eigen_payloads: list[dict[str, object]]) -> None:
    by_variant_key: dict[tuple[object, ...], list[dict[str, object]]] = {}
    by_layer_key: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for payload in eigen_payloads:
        by_variant_key.setdefault(_payload_group_key(payload, omit_variant=True), []).append(payload)
        by_layer_key.setdefault(
            (
                payload.get("model", ""),
                payload.get("variant", ""),
                payload.get("dataset_condition", ""),
                payload.get("revision", ""),
                payload.get("site", ""),
                payload.get("head", ""),
            ),
            [],
        ).append(payload)

    for key, payloads in by_variant_key.items():
        variants = sorted({str(payload.get("variant", "")) for payload in payloads})
        if len(variants) < 2:
            continue
        curves = [
            {"label": str(payload.get("variant", "")), "eigenvalues": payload["eigenvalues"]}
            for payload in sorted(payloads, key=lambda item: str(item.get("variant", "")))
        ]
        model, dataset, revision, layer, site, head = key
        head_part = f"_head{head}" if head != "" else ""
        stem = _safe_name(f"variant_overlay__{model}__{dataset}__{revision}__layer{layer}{head_part}__{site}")
        try:
            plot_eigenspectrum_overlay(
                curves,
                paths["comparisons"] / f"{stem}.png",
                title=f"{model} L{layer} {site}{head_part}: variant overlay",
                normalize=True,
            )
        except Exception as exc:
            LOGGER.warning("failed to plot variant overlay %s: %s", stem, exc)

    for key, payloads in by_layer_key.items():
        layers = sorted({int(payload.get("layer", -1)) for payload in payloads})
        if len(layers) < 2:
            continue
        curves = [
            {"label": f"L{payload.get('layer')}", "eigenvalues": payload["eigenvalues"]}
            for payload in sorted(payloads, key=lambda item: int(item.get("layer", -1)))
        ]
        model, variant, dataset, revision, site, head = key
        head_part = f"_head{head}" if head != "" else ""
        stem = _safe_name(f"layer_overlay__{model}__{variant}__{dataset}__{revision}{head_part}__{site}")
        try:
            plot_eigenspectrum_overlay(
                curves,
                paths["comparisons"] / f"{stem}.png",
                title=f"{model} {variant} {site}{head_part}: layer overlay",
                normalize=True,
            )
        except Exception as exc:
            LOGGER.warning("failed to plot layer overlay %s: %s", stem, exc)


def _write_metric_delta_tables(paths: dict[str, Path], rows: list[dict[str, object]]) -> None:
    metrics = {
        "effective_rank": "effective_rank_delta.csv",
        "top_1_explained_variance": "top1_delta.csv",
        "alpha": "alpha_delta.csv",
    }
    group_fields = ["model", "dataset_condition", "revision", "layer", "site", "head"]
    grouped: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        grouped.setdefault(key, {})[str(row.get("variant", ""))] = row
    for metric, filename in metrics.items():
        out_rows: list[dict[str, object]] = []
        for key, by_variant in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
            if "pretrained" not in by_variant or "random" not in by_variant:
                continue
            pretrained = _value_as_float(by_variant["pretrained"].get(metric))
            random = _value_as_float(by_variant["random"].get(metric))
            if pretrained is None or random is None:
                continue
            out = {field: value for field, value in zip(group_fields, key)}
            out.update(
                {
                    "metric": metric,
                    "pretrained": pretrained,
                    "random": random,
                    "random_minus_pretrained": random - pretrained,
                    "pretrained_minus_random": pretrained - random,
                }
            )
            out_rows.append(out)
        if not out_rows:
            out_rows.append(
                {
                    "metric": metric,
                    "status": "unavailable",
                    "reason": "requires matched pretrained and random variants in the same model/layer/site group",
                }
            )
        _write_csv(paths["metrics"] / filename, out_rows)


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    updates: dict[str, Any] = {}
    analysis_updates: dict[str, Any] = {}
    dynamic_updates: dict[str, Any] = {}
    if args.output_dir:
        updates["output_dir"] = args.output_dir
    if args.models:
        updates["models"] = parse_csv_arg(args.models)
    if args.variants:
        updates["variants"] = parse_csv_arg(args.variants)
    if args.sites:
        updates["sites"] = parse_csv_arg(args.sites)
    if args.layers:
        updates["layers"] = parse_layers(args.layers)
    if args.num_sequences is not None:
        updates["num_sequences"] = args.num_sequences
    if args.seq_len is not None:
        updates["sequence_length"] = args.seq_len
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.dataset_condition:
        updates["dataset_condition"] = args.dataset_condition
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.revision:
        updates["revision"] = args.revision
    if args.sample_limit is not None:
        analysis_updates["sample_limit"] = args.sample_limit
    if args.bootstrap_samples is not None:
        analysis_updates["bootstrap_samples"] = args.bootstrap_samples
    if args.powerlaw_rank_max is not None:
        analysis_updates["powerlaw_rank_max"] = args.powerlaw_rank_max
    if args.dynamic_max_sequences is not None:
        dynamic_updates["max_sequences"] = args.dynamic_max_sequences
    if args.dynamic_pca_rank is not None:
        dynamic_updates["pca_rank"] = args.dynamic_pca_rank
    if args.device_map:
        updates["device_map"] = args.device_map
    if args.torch_dtype:
        updates["torch_dtype"] = args.torch_dtype
    if args.local_files_only:
        updates["local_files_only"] = True
    if args.low_cpu_mem_usage:
        updates["low_cpu_mem_usage"] = True
    if args.no_plots:
        updates["plots"] = False
    if args.synthetic_smoke:
        updates["synthetic_smoke"] = True
    if args.fresh_output:
        updates["fresh_output"] = True
    if dynamic_updates:
        analysis_updates["dynamic"] = dynamic_updates
    if analysis_updates:
        updates["analysis"] = analysis_updates
    return deep_update(cfg, updates)


def _analysis_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    analysis = cfg.get("analysis", {})
    dynamic = analysis.get("dynamic", {})
    return {
        "output_zscore": bool(analysis.get("zscore", False)),
        "exclude_first_tokens": int(analysis.get("exclude_first_tokens", 0)),
        "sample_limit": int(analysis.get("sample_limit", 2048)),
        "powerlaw_rank_min": int(analysis.get("powerlaw_rank_min", 2)),
        "powerlaw_rank_max": analysis.get("powerlaw_rank_max"),
        "bootstrap_samples": int(analysis.get("bootstrap_samples", 200)),
        "dynamic_enabled": bool(dynamic.get("enabled", True)),
        "dynamic_site": str(dynamic.get("site", "resid_post")),
        "dynamic_layer": dynamic.get("layer", "last"),
        "dynamic_pca_rank": int(dynamic.get("pca_rank", 64)),
        "dynamic_max_sequences": int(dynamic.get("max_sequences", 64)),
        "dynamic_lags": [int(x) for x in dynamic.get("lags", [1, 2, 4, 8, 16, 32])],
    }


def run(cfg: dict[str, Any]) -> dict[str, Path]:
    logging.basicConfig(level=getattr(logging, str(cfg.get("log_level", "INFO")).upper()), format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(cfg.get("output_dir", "results"))
    if bool(cfg.get("fresh_output", False)):
        staged = _stage_existing_outputs_to_trash(output_dir)
        if staged is not None:
            LOGGER.info("staged existing managed outputs under %s", staged)
    paths = ensure_output_tree(output_dir)
    models = list(cfg.get("models", ["gpt2", "EleutherAI/pythia-70m"]))
    variants = list(cfg.get("variants", ["pretrained", "random"]))
    sites = list(cfg.get("sites", ["resid_post", "attn_out", "mlp_out"]))
    layers = cfg.get("layers", "all")
    seed = int(cfg.get("seed", 17))
    analysis = _analysis_cfg(cfg)

    all_metric_rows: list[dict[str, object]] = []
    all_dynamic_rows: list[dict[str, object]] = []
    all_payloads: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []

    for model_name in models:
        for variant in variants:
            LOGGER.info("collecting model=%s variant=%s", model_name, variant)
            if cfg.get("synthetic_smoke"):
                selected_layers = [0, 1, 2] if layers == "all" else [int(x) for x in layers]
                result = collect_synthetic_statistics(
                    model_name=model_name,
                    variant=variant,
                    sites=sites,
                    layers=selected_layers,
                    num_sequences=int(cfg.get("num_sequences", 32)),
                    seq_len=int(cfg.get("sequence_length", 32)),
                    seed=seed,
                    **analysis,
                )
            else:
                result = collect_activation_statistics(
                    model_name=model_name,
                    variant=variant,
                    sites=sites,
                    layers=layers,
                    num_sequences=int(cfg.get("num_sequences", 512)),
                    seq_len=int(cfg.get("sequence_length", 256)),
                    batch_size=int(cfg.get("batch_size", 2)),
                    dataset_condition=str(cfg.get("dataset_condition", "natural")),
                    seed=seed,
                    revision=cfg.get("revision"),
                    device=str(cfg.get("device", "auto")),
                    device_map=cfg.get("device_map"),
                    torch_dtype=cfg.get("torch_dtype"),
                    local_files_only=bool(cfg.get("local_files_only", False)),
                    low_cpu_mem_usage=bool(cfg.get("low_cpu_mem_usage", False)),
                    **analysis,
                )
            all_metric_rows.extend(result.metric_rows)
            all_dynamic_rows.extend(result.dynamic_rows)
            all_payloads.extend(result.eigen_payloads)
            metadata.append(result.metadata)
            _save_eigen_payloads(paths, result.eigen_payloads)

    _write_csv(paths["metrics"] / "spectral_metrics.csv", all_metric_rows)
    _write_csv(paths["metrics"] / "dynamic_metrics.csv", all_dynamic_rows)
    _write_metric_delta_tables(paths, all_metric_rows)
    (paths["root"] / "run_metadata.json").write_text(json.dumps(_jsonify(metadata), indent=2), encoding="utf-8")

    if bool(cfg.get("plots", True)):
        _plot_outputs(paths, all_metric_rows, all_payloads)
    write_markdown_report(all_metric_rows, all_dynamic_rows, paths["root"] / "report.md")
    LOGGER.info("wrote results to %s", paths["root"])
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM spectral dynamics experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models")
    parser.add_argument("--variants")
    parser.add_argument("--sites")
    parser.add_argument("--layers")
    parser.add_argument("--num-sequences", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dataset-condition")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--revision")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--powerlaw-rank-max", type=int)
    parser.add_argument("--dynamic-max-sequences", type=int)
    parser.add_argument("--dynamic-pca-rank", type=int)
    parser.add_argument("--device-map")
    parser.add_argument("--torch-dtype")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--fresh-output", action="store_true", help="Move existing managed output artifacts into output_dir/trash before running.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = _config_from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
