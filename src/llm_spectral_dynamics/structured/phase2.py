from __future__ import annotations

import argparse
import logging

import numpy as np

from .approximations import approximate_weight, approximation_error_row, svd_factors
from .data import load_model_and_tokenizer_from_config, load_texts_from_config, token_batches
from .metrics import relative_fro_error
from .residuals import build_residual
from .utils import ensure_phase_dirs, load_structured_config, parse_csv, parse_float_csv, parse_layers, select_layer_positions, write_csv
from .weights import iter_linear_layers, weight_to_numpy


LOGGER = logging.getLogger("structured.phase2")


def _linear_output(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def _collect_xy(model, tokenizer, refs, *, data_cfg: dict[str, object], device: str, sample_limit: int, seed: int):
    import torch

    rng = np.random.default_rng(seed)
    records = {ref.name: {"x": [], "y": [], "ref": ref} for ref in refs}
    handles = []

    def make_hook(name):
        def hook(_module, inputs, output):
            rec = records[name]
            current = sum(arr.shape[0] for arr in rec["x"])
            if current >= sample_limit:
                return
            x = inputs[0].detach().float().cpu().reshape(-1, inputs[0].shape[-1]).numpy()
            y = output.detach().float().cpu().reshape(-1, output.shape[-1]).numpy()
            take = min(x.shape[0], sample_limit - current)
            if x.shape[0] > take:
                idx = rng.choice(x.shape[0], size=take, replace=False)
                x = x[idx]
                y = y[idx]
            else:
                x = x[:take]
                y = y[:take]
            rec["x"].append(x)
            rec["y"].append(y)

        return hook

    for ref in refs:
        handles.append(ref.module.register_forward_hook(make_hook(ref.name)))
    try:
        texts = load_texts_from_config(data_cfg, limit=int(data_cfg.get("calibration_sequences", 32)) * 2)
        with torch.no_grad():
            for batch in token_batches(
                tokenizer,
                texts,
                sequence_length=int(data_cfg.get("sequence_length", 128)),
                batch_size=int(data_cfg.get("batch_size", 1)),
                limit=int(data_cfg.get("calibration_sequences", 32)),
            ):
                model(input_ids=batch.to(device))
                if all(sum(arr.shape[0] for arr in rec["x"]) >= sample_limit for rec in records.values()):
                    break
    finally:
        for handle in handles:
            handle.remove()
    out = {}
    for name, rec in records.items():
        if not rec["x"]:
            continue
        out[name] = {
            "ref": rec["ref"],
            "x": np.concatenate(rec["x"], axis=0)[:sample_limit],
            "y": np.concatenate(rec["y"], axis=0)[:sample_limit],
        }
    return out


def run_phase2(cfg: dict[str, object], *, max_modules: int | None = None) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = ensure_phase_dirs(str(cfg.get("output_dir", "results/structured_qwen25_1p5b")))
    model, tokenizer, device = load_model_and_tokenizer_from_config(cfg)
    refs_all = iter_linear_layers(model, list(cfg.get("target_modules", [])), cfg.get("layers", "all"))
    all_layers = sorted({ref.layer for ref in refs_all if ref.layer >= 0})
    phase2_cfg = dict(cfg.get("phase2", {}))
    if cfg.get("layers", "all") == "all":
        selected_layers = select_layer_positions(all_layers, list(phase2_cfg.get("layer_positions", ["first", "middle", "last"])))
    else:
        selected_layers = list(cfg.get("layers", []))
    refs = [ref for ref in refs_all if ref.layer in selected_layers]
    if max_modules is not None:
        refs = refs[: max(0, int(max_modules))]
    if not refs:
        raise RuntimeError("Phase 2 selected no linear modules; check target_modules, layers, and layer_positions")
    sample_limit = int(phase2_cfg.get("sample_limit", 2048))
    if sample_limit <= 0:
        raise ValueError("Phase 2 sample_limit must be positive")
    LOGGER.info("collecting activation reconstruction data for %d modules", len(refs))
    xy = _collect_xy(
        model,
        tokenizer,
        refs,
        data_cfg=dict(cfg.get("data", {})),
        device=device,
        sample_limit=sample_limit,
        seed=int(cfg.get("seed", 17)),
    )
    if not xy:
        raise RuntimeError("Phase 2 collected no activation records")
    approx_cfg = dict(cfg.get("approximation", {}))
    budget_cfg = dict(cfg.get("budgets", {}))
    ratios = [float(x) for x in budget_cfg.get("compression_ratios", [2, 4, 8, 16])]
    residual_fractions = [float(x) for x in budget_cfg.get("residual_fractions", [0.0, 0.01, 0.02, 0.04])]
    residual_types = [str(x) for x in approx_cfg.get("residual_types", ["none", "low_rank", "sparse", "channel"])]
    svd_device = str(approx_cfg.get("svd_device", "cpu"))
    methods = ["low_rank", "block_circulant", "monarch_like"]
    rows: list[dict[str, object]] = []
    for name, item in xy.items():
        ref = item["ref"]
        x = item["x"]
        y = item["y"]
        weight = weight_to_numpy(ref.module)
        weight_factors = svd_factors(weight, device=svd_device)
        bias = None
        if getattr(ref.module, "bias", None) is not None:
            bias = ref.module.bias.detach().float().cpu().numpy()
        base = {
            "name": name,
            "layer": ref.layer,
            "module_type": ref.module_type,
            "samples": int(x.shape[0]),
            "in_features": ref.in_features,
            "out_features": ref.out_features,
        }
        for ratio in ratios:
            for method in methods:
                approx = approximate_weight(
                    weight,
                    method=method,
                    compression_ratio=ratio,
                    block_sizes=[int(v) for v in approx_cfg.get("block_sizes", [16, 32, 64, 128])],
                    monarch_block_size=int(approx_cfg.get("monarch_block_size", 64)),
                    monarch_terms=int(approx_cfg.get("monarch_terms", 2)),
                    low_rank_factors=weight_factors,
                    svd_device=svd_device,
                )
                y_hat = _linear_output(x, approx.matrix, bias)
                row_base = dict(base)
                row_base.update(approximation_error_row(weight, approx, compression_ratio=ratio))
                row_base["relative_activation_error"] = relative_fro_error(y, y_hat)
                residual = weight - approx.matrix
                residual_factors = (
                    svd_factors(residual, device=svd_device)
                    if "low_rank" in residual_types and any(float(frac) > 0 for frac in residual_fractions)
                    else None
                )
                for frac in residual_fractions:
                    types_for_fraction = ["none"] if float(frac) <= 0 else residual_types
                    for residual_type in types_for_fraction:
                        rr = build_residual(
                            residual,
                            residual_type=residual_type,
                            residual_fraction=frac,
                            low_rank_factors=residual_factors,
                        )
                        y_res = _linear_output(x, approx.matrix + rr.matrix, bias)
                        rows.append(
                            {
                                **row_base,
                                "residual_fraction": float(frac),
                                "residual_type": rr.residual_type,
                                "residual_params": int(rr.params),
                                "residual_rank": int(rr.rank),
                                "residual_channels": int(rr.channels),
                                "relative_activation_error_after_residual": relative_fro_error(y, y_res),
                                "relative_weight_error_after_residual": relative_fro_error(weight, approx.matrix + rr.matrix),
                            }
                        )
    if not rows:
        raise RuntimeError("Phase 2 produced no activation reconstruction rows")
    write_csv(paths["phase2_metrics"] / "activation_reconstruction.csv", rows)
    return {"rows": len(rows), "modules": len(xy)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 single-layer activation reconstruction.")
    parser.add_argument("--config", default="configs/structured_qwen25_1p5b.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--model")
    parser.add_argument("--layers")
    parser.add_argument("--modules")
    parser.add_argument("--compression-ratios")
    parser.add_argument("--residual-fractions")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--calibration-sequences", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-modules", type=int)
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
    if args.sample_limit is not None:
        overrides["phase2"] = {"sample_limit": args.sample_limit}
    data_updates: dict[str, object] = {}
    if args.calibration_sequences is not None:
        data_updates["calibration_sequences"] = args.calibration_sequences
    if data_updates:
        overrides["data"] = data_updates
    budget_updates: dict[str, object] = {}
    if args.compression_ratios:
        budget_updates["compression_ratios"] = parse_float_csv(args.compression_ratios)
    if args.residual_fractions:
        budget_updates["residual_fractions"] = parse_float_csv(args.residual_fractions)
    if budget_updates:
        overrides["budgets"] = budget_updates
    return load_structured_config(args.config, overrides)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_phase2(config_from_args(args), max_modules=args.max_modules)


if __name__ == "__main__":
    main()
