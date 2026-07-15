from __future__ import annotations

import argparse
import logging

import numpy as np

from .approximations import approximate_weight, approximation_error_row, batched_svd_factors, svd_factors
from .data import load_model_and_tokenizer_from_config
from .metrics import residual_distribution_metrics, weight_spectrum_metrics
from .residuals import residual_analysis_rows
from .utils import ensure_phase_dirs, load_structured_config, parse_csv, parse_float_csv, parse_layers, write_csv, write_json
from .weights import iter_linear_layers, weight_to_numpy


LOGGER = logging.getLogger("structured.phase1")


def run_phase1(cfg: dict[str, object], *, max_matrices: int | None = None) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = ensure_phase_dirs(str(cfg.get("output_dir", "results/structured_qwen25_1p5b")))
    model, _tokenizer, _device = load_model_and_tokenizer_from_config(cfg)
    target_modules = list(cfg.get("target_modules", []))
    layers = cfg.get("layers", "all")
    refs = iter_linear_layers(model, target_modules, layers)
    if max_matrices is not None:
        refs = refs[: max(0, int(max_matrices))]
    if not refs:
        raise RuntimeError("Phase 1 selected no linear weights; check target_modules and layers")
    approx_cfg = dict(cfg.get("approximation", {}))
    budget_cfg = dict(cfg.get("budgets", {}))
    ratios = [float(x) for x in budget_cfg.get("compression_ratios", [2, 4, 8, 16])]
    residual_fractions = [float(x) for x in budget_cfg.get("residual_fractions", [0.0, 0.01, 0.02, 0.04])]
    residual_types = [str(x) for x in approx_cfg.get("residual_types", ["none", "low_rank", "sparse", "channel"])]
    methods = ["low_rank", "block_circulant", "monarch_like"]

    layer_rows: list[dict[str, object]] = []
    spectra_rows: list[dict[str, object]] = []
    approx_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []

    for idx, ref in enumerate(refs, start=1):
        LOGGER.info("phase1 %d/%d %s", idx, len(refs), ref.name)
        weight = weight_to_numpy(ref.module)
        svd_device = str(approx_cfg.get("svd_device", "cpu"))
        factors = svd_factors(weight, device=svd_device)
        s = factors[1].astype("float64")
        base = {
            "name": ref.name,
            "layer": ref.layer,
            "module_type": ref.module_type,
            "in_features": ref.in_features,
            "out_features": ref.out_features,
            "has_bias": ref.has_bias,
        }
        metrics = dict(base)
        metrics.update(weight_spectrum_metrics(weight, s))
        layer_rows.append(metrics)
        total = float((s * s).sum())
        for rank_idx, sigma in enumerate(s[: min(512, s.size)], start=1):
            spectra_rows.append(
                {
                    **base,
                    "rank_index": rank_idx,
                    "singular_value": float(sigma),
                    "energy_fraction": float((sigma * sigma) / max(total, 1e-30)),
                }
            )
        residual_work: list[tuple[float, object, tuple[float, str]]] = []
        for ratio in ratios:
            best_by_error: tuple[float, str] | None = None
            best_structured = None
            for method in methods:
                result = approximate_weight(
                    weight,
                    method=method,
                    compression_ratio=ratio,
                    block_sizes=[int(x) for x in approx_cfg.get("block_sizes", [16, 32, 64, 128])],
                    monarch_block_size=int(approx_cfg.get("monarch_block_size", 64)),
                    monarch_terms=int(approx_cfg.get("monarch_terms", 2)),
                    low_rank_factors=factors,
                    svd_device=svd_device,
                )
                row = dict(base)
                row.update(approximation_error_row(weight, result, compression_ratio=ratio))
                approx_rows.append(row)
                error = float(row["relative_weight_error"])
                if best_by_error is None or error < best_by_error[0]:
                    best_by_error = (error, method)
                    best_structured = result.matrix
            if best_structured is not None and best_by_error is not None:
                residual_work.append((ratio, best_structured, best_by_error))

        residual_svd_batch_size = max(1, int(approx_cfg.get("residual_svd_batch_size", 1)))
        for start in range(0, len(residual_work), residual_svd_batch_size):
            chunk = residual_work[start : start + residual_svd_batch_size]
            residual_matrices = [weight - structured for _ratio, structured, _best in chunk]
            if len(chunk) > 1:
                u_batch, s_batch, vh_batch = batched_svd_factors(np.stack(residual_matrices), device=svd_device)
                factor_sets = [(u_batch[i], s_batch[i], vh_batch[i]) for i in range(len(chunk))]
            else:
                factor_sets = [svd_factors(residual_matrices[0], device=svd_device)]
            for (ratio, best_structured, best_by_error), residual, residual_factors in zip(chunk, residual_matrices, factor_sets):
                for row in residual_analysis_rows(
                    weight,
                    best_structured,
                    compression_ratio=ratio,
                    residual_fractions=residual_fractions,
                    residual_types=residual_types,
                    svd_device=svd_device,
                    residual_factors=residual_factors,
                ):
                    residual_rows.append({**base, **row})
                res_summary = residual_distribution_metrics(residual)
                manifest.append({**base, "compression_ratio_target": ratio, "best_method": best_by_error[1], **res_summary})

    write_csv(paths["phase1_metrics"] / "layer_spectrum_metrics.csv", layer_rows)
    write_csv(paths["phase1_metrics"] / "spectra.csv", spectra_rows)
    write_csv(paths["phase1_metrics"] / "approximation_errors.csv", approx_rows)
    write_csv(paths["phase1_metrics"] / "residual_metrics.csv", residual_rows)
    write_json(paths["manifests"] / "phase1_manifest.json", {"model": cfg["model"], "layers": layers, "modules": target_modules, "rows": manifest})
    return {"layer_rows": len(layer_rows), "approximation_rows": len(approx_rows), "residual_rows": len(residual_rows)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1 structured weight diagnostics.")
    parser.add_argument("--config", default="configs/structured_qwen25_1p5b.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--model")
    parser.add_argument("--layers")
    parser.add_argument("--modules")
    parser.add_argument("--compression-ratios")
    parser.add_argument("--residual-fractions")
    parser.add_argument("--residual-svd-batch-size", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-matrices", type=int)
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.model:
        overrides["model"] = args.model
    if args.layers:
        overrides["layers"] = parse_layers(args.layers)
    if args.modules:
        overrides["target_modules"] = parse_csv(args.modules)
    if args.local_files_only:
        overrides["local_files_only"] = True
    budget_updates: dict[str, object] = {}
    if args.compression_ratios:
        budget_updates["compression_ratios"] = parse_float_csv(args.compression_ratios)
    if args.residual_fractions:
        budget_updates["residual_fractions"] = parse_float_csv(args.residual_fractions)
    if budget_updates:
        overrides["budgets"] = budget_updates
    if args.residual_svd_batch_size is not None:
        overrides["approximation"] = {"residual_svd_batch_size": max(1, int(args.residual_svd_batch_size))}
    return load_structured_config(args.config, overrides)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_phase1(config_from_args(args), max_matrices=args.max_matrices)


if __name__ == "__main__":
    main()
