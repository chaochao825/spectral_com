from __future__ import annotations

import argparse
import logging

from .data import load_model_and_tokenizer_from_config
from .quantization import quantization_error_rows, structured_quantization_rows
from .rotation import butterfly_rotate_columns, hadamard_rotate_columns, learned_butterfly_rotate_columns, rotation_metrics
from .utils import ensure_phase_dirs, load_structured_config, parse_csv, parse_layers, select_layer_positions, write_csv
from .weights import iter_linear_layers, weight_to_numpy


LOGGER = logging.getLogger("structured.phase5")


def _select_refs(model, cfg: dict[str, object], *, max_matrices: int | None):
    phase_cfg = dict(cfg.get("phase5", {}))
    target_modules = [str(x) for x in phase_cfg.get("target_modules", cfg.get("target_modules", []))]
    refs_all = iter_linear_layers(model, target_modules, cfg.get("layers", "all"))
    all_layers = sorted({ref.layer for ref in refs_all if ref.layer >= 0})
    if cfg.get("layers", "all") == "all":
        selected_layers = select_layer_positions(all_layers, list(phase_cfg.get("layer_positions", ["first", "middle", "last"])))
    else:
        selected_layers = list(cfg.get("layers", []))
    refs = [ref for ref in refs_all if ref.layer in selected_layers and ref.module_type in target_modules]
    if max_matrices is not None:
        refs = refs[: max(0, int(max_matrices))]
    return refs


def _rotate(weight, rotation_type: str, phase_cfg: dict[str, object]):
    if rotation_type == "none":
        return weight
    if rotation_type == "hadamard":
        return hadamard_rotate_columns(weight)
    if rotation_type == "learned_butterfly":
        return learned_butterfly_rotate_columns(
            weight,
            steps=int(phase_cfg.get("learned_rotation_steps", 16)),
            lr=float(phase_cfg.get("learned_rotation_lr", 0.05)),
        )
    if rotation_type == "butterfly":
        return butterfly_rotate_columns(weight)
    raise ValueError(f"unknown rotation type: {rotation_type}")


def run_phase5(cfg: dict[str, object], *, max_matrices: int | None = None) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = ensure_phase_dirs(str(cfg.get("output_dir", "results/structured_qwen25_1p5b")))
    phase_cfg = dict(cfg.get("phase5", {}))
    approx_cfg = dict(cfg.get("approximation", {}))
    budget_cfg = dict(cfg.get("budgets", {}))
    bit_widths = [int(x) for x in phase_cfg.get("bit_widths", [4, 3, 2])]
    rotations = [str(x) for x in phase_cfg.get("rotation_types", ["none", "hadamard", "learned_butterfly"])]
    ratios = [float(x) for x in budget_cfg.get("compression_ratios", [4])]
    residual_fractions = [float(x) for x in budget_cfg.get("residual_fractions", [0.0, 0.02])]
    residual_precision = str(phase_cfg.get("residual_precision", "float32"))
    residual_type = str(approx_cfg.get("residual_types", ["low_rank"])[1] if len(approx_cfg.get("residual_types", [])) > 1 else "low_rank")
    model, _tokenizer, _device = load_model_and_tokenizer_from_config(cfg)
    refs = _select_refs(model, cfg, max_matrices=max_matrices or phase_cfg.get("max_matrices"))
    rotation_rows: list[dict[str, object]] = []
    quant_rows: list[dict[str, object]] = []
    structured_rows: list[dict[str, object]] = []
    for idx, ref in enumerate(refs, start=1):
        LOGGER.info("phase5 %d/%d %s", idx, len(refs), ref.name)
        weight = weight_to_numpy(ref.module)
        base = {"name": ref.name, "layer": ref.layer, "module_type": ref.module_type, "rows": ref.out_features, "cols": ref.in_features}
        for rotation_type in rotations:
            rotated = _rotate(weight, rotation_type, phase_cfg)
            rotation_rows.append({**base, **rotation_metrics(weight, rotated, rotation_type=rotation_type)})
            for row in quantization_error_rows(rotated, bit_widths=bit_widths, prefix="rotated_weight"):
                quant_rows.append({**base, "rotation_type": rotation_type, **row})
            for ratio in ratios:
                for residual_fraction in residual_fractions:
                    for method in ["low_rank", "block_circulant", "monarch_like"]:
                        for row in structured_quantization_rows(
                            rotated,
                            compression_ratio=ratio,
                            method=method,
                            bit_widths=bit_widths,
                            residual_fraction=residual_fraction,
                            residual_type=residual_type,
                            block_sizes=[int(v) for v in approx_cfg.get("block_sizes", [16, 32, 64, 128])],
                            monarch_block_size=int(approx_cfg.get("monarch_block_size", 64)),
                            monarch_terms=int(approx_cfg.get("monarch_terms", 2)),
                            residual_precision=residual_precision,
                            svd_device=str(approx_cfg.get("svd_device", "cpu")),
                        ):
                            structured_rows.append({**base, "rotation_type": rotation_type, **row})
    write_csv(paths["phase5_metrics"] / "rotation_outliers.csv", rotation_rows)
    write_csv(paths["phase5_metrics"] / "quantization_errors.csv", quant_rows)
    write_csv(paths["phase5_metrics"] / "structured_quantization.csv", structured_rows)
    return {"rotation_rows": len(rotation_rows), "quantization_rows": len(quant_rows), "structured_rows": len(structured_rows)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5 rotation and quantization analysis.")
    parser.add_argument("--config", default="configs/structured_qwen25_1p5b.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--rotations")
    parser.add_argument("--layers")
    parser.add_argument("--modules")
    parser.add_argument("--bit-widths")
    parser.add_argument("--max-matrices", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.local_files_only:
        overrides["local_files_only"] = True
    if args.layers:
        overrides["layers"] = parse_layers(args.layers)
    phase_updates: dict[str, object] = {}
    if args.modules:
        phase_updates["target_modules"] = parse_csv(args.modules)
    if args.rotations:
        phase_updates["rotation_types"] = parse_csv(args.rotations)
    if args.bit_widths:
        phase_updates["bit_widths"] = [int(x) for x in parse_csv(args.bit_widths) or []]
    if args.max_matrices is not None:
        phase_updates["max_matrices"] = int(args.max_matrices)
    if phase_updates:
        overrides["phase5"] = phase_updates
    return load_structured_config(args.config, overrides)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_phase5(config_from_args(args), max_matrices=args.max_matrices)


if __name__ == "__main__":
    main()
