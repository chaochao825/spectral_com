from __future__ import annotations

import argparse
import csv
import logging
import statistics
from collections import defaultdict
from pathlib import Path

from .data import load_model_and_tokenizer_from_config
from .evaluation import evaluate_perplexity, evaluate_zero_shot
from .replacement import replace_model_linears
from .utils import ensure_phase_dirs, load_structured_config, parse_csv, parse_float_csv, parse_layers, write_csv, write_json


LOGGER = logging.getLogger("structured.phase3")
PHASE1_APPROXIMATION_METHODS = ("low_rank", "block_circulant", "monarch_like")


def _best_methods_from_phase1(output_dir: str | Path, ratio: float) -> dict[str, str]:
    path = Path(output_dir) / "phase1" / "metrics" / "approximation_errors.csv"
    if not path.exists():
        return {}
    errors: dict[tuple[str, str], dict[tuple[str, int], float]] = defaultdict(dict)
    module_types: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                row_ratio = float(row.get("compression_ratio_target", "nan"))
                if abs(row_ratio - float(ratio)) > 1e-9:
                    continue
                module_type = str(row["module_type"])
                method = str(row["method"])
                name = str(row["name"])
                layer = int(row["layer"])
                error = float(row["relative_weight_error"])
            except Exception:
                continue
            module_types.add(module_type)
            key = (name, layer)
            if key in errors[(module_type, method)]:
                raise ValueError(f"Phase 1 approximation errors contain duplicate rows for {module_type}/{method}: {key}")
            errors[(module_type, method)][key] = error

    best: dict[str, tuple[float, str]] = {}
    for module_type in sorted(module_types):
        missing_methods = [method for method in PHASE1_APPROXIMATION_METHODS if not errors[(module_type, method)]]
        if missing_methods:
            raise ValueError(f"Phase 1 approximation errors are missing methods for {module_type}: {missing_methods}")
        expected_coverage = set(errors[(module_type, PHASE1_APPROXIMATION_METHODS[0])])
        for method in PHASE1_APPROXIMATION_METHODS[1:]:
            if set(errors[(module_type, method)]) != expected_coverage:
                raise ValueError(f"Phase 1 approximation errors have inconsistent layer coverage for {module_type}")
        for method in PHASE1_APPROXIMATION_METHODS:
            values = list(errors[(module_type, method)].values())
            error = float(statistics.median(values))
            current = best.get(module_type)
            if current is None or (error, method) < current:
                best[module_type] = (error, method)
    return {key: value[1] for key, value in best.items()}


def _method_mapping(cfg: dict[str, object], ratio: float) -> dict[str, str]:
    phase3_cfg = dict(cfg.get("phase3", {}))
    structure = str(phase3_cfg.get("structure", "low_rank"))
    if structure == "best_weight_error":
        methods = _best_methods_from_phase1(str(cfg.get("output_dir", "results/structured_qwen25_1p5b")), ratio)
        required = [str(module) for module in cfg.get("target_modules", [])]
        missing = [module for module in required if module not in methods]
        if missing:
            raise ValueError(
                f"Phase 1 approximation errors are missing best_weight_error selections "
                f"for ratio={ratio}: {missing}"
            )
        return methods
    return {str(module): structure for module in cfg.get("target_modules", [])}


def _total_record_params(records) -> tuple[int, int, int]:
    original = sum(int(item.params_original) for item in records)
    structured = sum(int(item.params_structured) for item in records)
    residual = sum(int(item.params_residual) for item in records)
    return original, structured, residual


def _stage_modules(stage: dict[str, object], target_modules: list[str]) -> list[str]:
    target = set(target_modules)
    return [str(module) for module in stage.get("modules", []) if str(module) in target]


def _append_replaced_modules(current: list[str], records) -> list[str]:
    out = list(current)
    for record in records:
        module_type = str(record.module_type)
        if module_type not in out:
            out.append(module_type)
    return out


def run_phase3(cfg: dict[str, object], *, skip_zero_shot: bool = False) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = ensure_phase_dirs(str(cfg.get("output_dir", "results/structured_qwen25_1p5b")))
    budget_cfg = dict(cfg.get("budgets", {}))
    ratios = [float(x) for x in budget_cfg.get("compression_ratios", [2, 4, 8, 16])]
    residual_fractions = [float(x) for x in budget_cfg.get("residual_fractions", [0.0, 0.01, 0.02, 0.04])]
    approx_cfg = dict(cfg.get("approximation", {}))
    phase3_cfg = dict(cfg.get("phase3", {}))
    stages = list(phase3_cfg.get("stages", []))
    perf_rows: list[dict[str, object]] = []
    zero_rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []

    LOGGER.info("loading baseline model")
    baseline_model, baseline_tokenizer, baseline_device = load_model_and_tokenizer_from_config(cfg)
    baseline = evaluate_perplexity(
        baseline_model,
        baseline_tokenizer,
        data_cfg=dict(cfg.get("data", {})),
        device=baseline_device,
        eval_limit=int(phase3_cfg.get("eval_limit", 32)),
    )
    perf_rows.append({"stage": "baseline", "compression_ratio_target": 1.0, "residual_fraction": 0.0, **baseline})
    if not skip_zero_shot:
        for row in evaluate_zero_shot(
            baseline_model,
            baseline_tokenizer,
            tasks=[str(x) for x in phase3_cfg.get("zero_shot_tasks", [])],
            limit=int(phase3_cfg.get("zero_shot_limit", 16)),
            device=baseline_device,
        ):
            zero_rows.append({"stage": "baseline", "compression_ratio_target": 1.0, "residual_fraction": 0.0, **row})
    del baseline_model

    for ratio in ratios:
        for residual_fraction in residual_fractions:
            LOGGER.info("phase3 ratio=%s residual_fraction=%s", ratio, residual_fraction)
            model, tokenizer, device = load_model_and_tokenizer_from_config(cfg)
            cumulative_modules: list[str] = []
            all_records = []
            method_by_module = _method_mapping(cfg, ratio)
            target_modules = [str(x) for x in cfg.get("target_modules", [])]
            for stage in stages:
                stage_name = str(stage["name"])
                stage_modules = _stage_modules(stage, target_modules)
                if not stage_modules:
                    LOGGER.info("skipping stage=%s because no stage modules are in target_modules", stage_name)
                    continue
                records = replace_model_linears(
                    model,
                    target_modules=target_modules,
                    layers=cfg.get("layers", "all"),
                    module_types_to_replace=stage_modules,
                    method_by_module=method_by_module,
                    compression_ratio=ratio,
                    residual_type=str(phase3_cfg.get("residual_type", "low_rank")),
                    residual_fraction=residual_fraction,
                    block_sizes=[int(v) for v in approx_cfg.get("block_sizes", [16, 32, 64, 128])],
                    monarch_block_size=int(approx_cfg.get("monarch_block_size", 64)),
                    monarch_terms=int(approx_cfg.get("monarch_terms", 2)),
                    svd_device=str(approx_cfg.get("svd_device", "cpu")),
                )
                if not records:
                    LOGGER.info("skipping stage=%s because no modules were actually replaced", stage_name)
                    continue
                all_records.extend(records)
                cumulative_modules = _append_replaced_modules(cumulative_modules, records)
                original, structured, residual = _total_record_params(all_records)
                scores = evaluate_perplexity(
                    model,
                    tokenizer,
                    data_cfg=dict(cfg.get("data", {})),
                    device=device,
                    eval_limit=int(phase3_cfg.get("eval_limit", 32)),
                )
                perf_rows.append(
                    {
                        "stage": stage_name,
                        "modules_replaced": ",".join(cumulative_modules),
                        "compression_ratio_target": ratio,
                        "residual_fraction": residual_fraction,
                        "params_original_replaced": original,
                        "params_structured_replaced": structured,
                        "params_residual_replaced": residual,
                        "effective_replaced_compression": original / max(structured + residual, 1),
                        **scores,
                    }
                )
                if not skip_zero_shot:
                    for row in evaluate_zero_shot(
                        model,
                        tokenizer,
                        tasks=[str(x) for x in phase3_cfg.get("zero_shot_tasks", [])],
                        limit=int(phase3_cfg.get("zero_shot_limit", 16)),
                        device=device,
                    ):
                        zero_rows.append({"stage": stage_name, "compression_ratio_target": ratio, "residual_fraction": residual_fraction, **row})
            manifests.append(
                {
                    "compression_ratio_target": ratio,
                    "residual_fraction": residual_fraction,
                    "methods": method_by_module,
                    "records": [item.__dict__ for item in all_records],
                }
            )
            del model

    write_csv(paths["phase3_metrics"] / "compression_performance.csv", perf_rows)
    write_csv(paths["phase3_metrics"] / "zero_shot.csv", zero_rows)
    write_json(paths["manifests"] / "phase3_replacements.json", manifests)
    _write_report(paths["root"], perf_rows, zero_rows)
    return {"performance_rows": len(perf_rows), "zero_shot_rows": len(zero_rows)}


def _write_report(root: Path, perf_rows: list[dict[str, object]], zero_rows: list[dict[str, object]]) -> None:
    lines = ["# Structured Qwen2.5-1.5B Compression Report", ""]
    baseline = next((row for row in perf_rows if row.get("stage") == "baseline"), None)
    if baseline:
        lines.append(f"- Baseline perplexity: {float(baseline['perplexity']):.4g}.")
    completed = [row for row in perf_rows if row.get("stage") != "baseline"]
    if completed:
        best = min(completed, key=lambda row: float(row.get("perplexity", float("inf"))))
        lines.append(
            "- Best compressed PPL row: "
            f"stage={best.get('stage')}, ratio={best.get('compression_ratio_target')}, "
            f"residual={best.get('residual_fraction')}, ppl={float(best['perplexity']):.4g}."
        )
        by_stage = sorted({str(row.get("stage")) for row in completed})
        lines.append(f"- Completed replacement stages: {', '.join(by_stage)}.")
    if zero_rows:
        ok_rows = [row for row in zero_rows if row.get("status") == "ok"]
        if ok_rows:
            mean_acc = sum(float(row["accuracy"]) for row in ok_rows) / len(ok_rows)
            lines.append(f"- Mean reported zero-shot accuracy across completed rows: {mean_acc:.4g}.")
        else:
            lines.append("- Zero-shot rows were unavailable; inspect `phase3/metrics/zero_shot.csv` for task errors.")
    lines.append("")
    lines.append("Detailed CSV artifacts are under `phase1/metrics`, `phase2/metrics`, and `phase3/metrics`.")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 model-level post-training compression.")
    parser.add_argument("--config", default="configs/structured_qwen25_1p5b.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--model")
    parser.add_argument("--layers")
    parser.add_argument("--modules")
    parser.add_argument("--compression-ratios")
    parser.add_argument("--residual-fractions")
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--zero-shot-limit", type=int)
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
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
    phase3_updates: dict[str, object] = {}
    if args.eval_limit is not None:
        phase3_updates["eval_limit"] = args.eval_limit
    if args.zero_shot_limit is not None:
        phase3_updates["zero_shot_limit"] = args.zero_shot_limit
    if phase3_updates:
        overrides["phase3"] = phase3_updates
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
    run_phase3(config_from_args(args), skip_zero_shot=args.skip_zero_shot)


if __name__ == "__main__":
    main()
