from __future__ import annotations

import argparse
import logging

import numpy as np

from .adapters import AdapterWrappedLinear
from .data import load_model_and_tokenizer_from_config, load_texts_from_config, token_batches
from .evaluation import evaluate_perplexity
from .metrics import singular_values, weight_spectrum_metrics
from .utils import ensure_phase_dirs, load_structured_config, parse_csv, parse_layers, select_layer_positions, set_global_seed, write_csv, write_json
from .weights import get_submodule_parent, iter_linear_layers


LOGGER = logging.getLogger("structured.phase4")

STRUCTURED_TASK_TEXTS = [
    "Question: Paris is in which country?\nAnswer: France\nQuestion: Rome is in which country?\nAnswer: Italy",
    "copy: red blue green red blue green red blue green",
    "1 + 1 = 2\n2 + 2 = 4\n3 + 3 = 6\n4 + 4 = 8",
]


def _texts_for_condition(data_cfg: dict[str, object], condition: str, limit: int) -> list[str]:
    if condition == "structured":
        reps = (max(limit, 1) + len(STRUCTURED_TASK_TEXTS) - 1) // len(STRUCTURED_TASK_TEXTS)
        return (STRUCTURED_TASK_TEXTS * reps)[: max(limit, 1)]
    return load_texts_from_config(data_cfg, limit=limit)


def _select_refs(model, cfg: dict[str, object], target_modules: list[str], layer_positions: list[str | int], *, max_modules: int | None):
    refs_all = iter_linear_layers(model, target_modules, cfg.get("layers", "all"))
    all_layers = sorted({ref.layer for ref in refs_all if ref.layer >= 0})
    if cfg.get("layers", "all") == "all":
        selected_layers = select_layer_positions(all_layers, layer_positions)
    else:
        selected_layers = list(cfg.get("layers", []))
    refs = [ref for ref in refs_all if ref.layer in selected_layers and ref.module_type in target_modules]
    if max_modules is not None:
        refs = refs[: max(0, int(max_modules))]
    return refs


def _freeze_model(model) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _install_adapters(model, refs, *, method: str, budget: int, rank: int | None, block_size: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for ref in refs:
        parent, child_name = get_submodule_parent(model, ref.name)
        adapter = AdapterWrappedLinear(ref.module, method=method, budget=int(budget), rank=rank, block_size=block_size)
        wrapped = adapter.to_module()
        setattr(parent, child_name, wrapped)
        records.append(
            {
                "name": ref.name,
                "layer": ref.layer,
                "module_type": ref.module_type,
                "method": method,
                "budget_per_module": int(budget),
                "requested_rank": "" if rank is None else int(rank),
                "rank": "" if adapter.spec.rank <= 0 else int(adapter.spec.rank),
                "block_size": int(adapter.spec.block_size),
                "within_budget": bool(adapter.spec.within_budget),
                "adapter_params": sum(int(param.numel()) for param in wrapped.update_modules.parameters()),
            }
        )
    return records


def _train_adapters(model, tokenizer, *, data_cfg: dict[str, object], device: str, condition: str, train_steps: int, lr: float) -> list[float]:
    import torch

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        return []
    optimizer = torch.optim.AdamW(params, lr=float(lr))
    texts = _texts_for_condition(data_cfg, condition, limit=max(int(train_steps), 1) * 2)
    losses: list[float] = []
    batches = list(
        token_batches(
            tokenizer,
            texts,
            sequence_length=int(data_cfg.get("sequence_length", 128)),
            batch_size=int(data_cfg.get("batch_size", 1)),
            limit=max(int(train_steps), 1),
        )
    )
    model.train()
    for step in range(max(int(train_steps), 1)):
        batch = batches[step % len(batches)].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=batch)
        logits = outputs.logits[:, :-1, :].float()
        labels = batch[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return losses


def _iter_adapter_delta_weights(model):
    for name, module in model.named_modules():
        if hasattr(module, "delta_weight"):
            yield name, module.delta_weight().detach().float().cpu().numpy()


def run_phase4(cfg: dict[str, object], *, max_runs: int | None = None, max_modules: int | None = None) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = ensure_phase_dirs(str(cfg.get("output_dir", "results/structured_qwen25_1p5b")))
    phase_cfg = dict(cfg.get("phase4", {}))
    data_cfg = dict(cfg.get("data", {}))
    methods = [str(x) for x in phase_cfg.get("methods", ["structured", "lora"])]
    budgets = [int(x) for x in phase_cfg.get("parameter_budgets", [65536])]
    task_conditions = [str(x) for x in phase_cfg.get("task_conditions", ["natural"])]
    rank_sensitivity = [int(x) for x in phase_cfg.get("rank_sensitivity", [int(phase_cfg.get("lora_rank", 8))])]
    target_modules = [str(x) for x in phase_cfg.get("target_modules", cfg.get("target_modules", []))]
    runs: list[tuple[str, int, str, int | None]] = []
    for method in methods:
        for budget in budgets:
            for condition in task_conditions:
                ranks = rank_sensitivity if method in {"lora", "structured_lora", "mora"} else [None]
                for rank in ranks:
                    runs.append((method, budget, condition, rank))
    if max_runs is not None:
        runs = runs[: max(0, int(max_runs))]

    perf_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    base_seed = int(cfg.get("seed", 17))
    for idx, (method, budget, condition, rank) in enumerate(runs, start=1):
        run_seed = base_seed + idx - 1
        set_global_seed(run_seed)
        LOGGER.info("phase4 run %d/%d method=%s budget=%s condition=%s rank=%s", idx, len(runs), method, budget, condition, rank)
        model, tokenizer, device = load_model_and_tokenizer_from_config(cfg)
        _freeze_model(model)
        refs = _select_refs(model, cfg, target_modules, [x for x in phase_cfg.get("layer_positions", ["first"])], max_modules=max_modules)
        records = _install_adapters(
            model,
            refs,
            method=method,
            budget=budget,
            rank=rank,
            block_size=int(cfg.get("approximation", {}).get("monarch_block_size", 64)),
        )
        losses = _train_adapters(
            model,
            tokenizer,
            data_cfg=data_cfg,
            device=device,
            condition=condition,
            train_steps=int(phase_cfg.get("train_steps", 40)),
            lr=float(phase_cfg.get("learning_rate", 5e-4)),
        )
        scores = evaluate_perplexity(model, tokenizer, data_cfg=data_cfg, device=device, eval_limit=int(phase_cfg.get("eval_limit", 16)))
        adapter_params = sum(int(record["adapter_params"]) for record in records)
        actual_ranks = sorted({int(record["rank"]) for record in records if record["rank"] != ""})
        perf_rows.append(
            {
                "method": method,
                "seed": run_seed,
                "budget_per_module": int(budget),
                "requested_rank": "" if rank is None else int(rank),
                "rank": actual_ranks[0] if len(actual_ranks) == 1 else "",
                "task_condition": condition,
                "train_steps": int(phase_cfg.get("train_steps", 40)),
                "train_loss_first": losses[0] if losses else "",
                "train_loss_last": losses[-1] if losses else "",
                "adapter_params": adapter_params,
                "within_budget": all(bool(record["within_budget"]) for record in records),
                "modules": ",".join(str(record["module_type"]) for record in records),
                **scores,
            }
        )
        for name, delta in _iter_adapter_delta_weights(model):
            s = singular_values(delta)
            row = {
                "adapter_module": name,
                "method": method,
                "budget_per_module": int(budget),
                "requested_rank": "" if rank is None else int(rank),
                "rank": actual_ranks[0] if len(actual_ranks) == 1 else "",
                "task_condition": condition,
            }
            row.update(weight_spectrum_metrics(delta, s))
            spectrum_rows.append(row)
        manifests.append(
            {
                "method": method,
                "seed": run_seed,
                "budget_per_module": budget,
                "condition": condition,
                "requested_rank": rank,
                "records": records,
            }
        )
        del model

    write_csv(paths["phase4_metrics"] / "peft_performance.csv", perf_rows)
    write_csv(paths["phase4_metrics"] / "update_spectrum.csv", spectrum_rows)
    write_json(paths["manifests"] / "phase4_adapters.json", manifests)
    return {"performance_rows": len(perf_rows), "spectrum_rows": len(spectrum_rows)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4 structured PEFT validation.")
    parser.add_argument("--config", default="configs/structured_qwen25_1p5b.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--methods")
    parser.add_argument("--layers")
    parser.add_argument("--modules")
    parser.add_argument("--budgets")
    parser.add_argument("--task-conditions")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--max-modules", type=int)
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
    if args.methods:
        phase_updates["methods"] = parse_csv(args.methods)
    if args.budgets:
        phase_updates["parameter_budgets"] = [int(x) for x in parse_csv(args.budgets) or []]
    if args.task_conditions:
        phase_updates["task_conditions"] = parse_csv(args.task_conditions)
    if args.train_steps is not None:
        phase_updates["train_steps"] = int(args.train_steps)
    if args.eval_limit is not None:
        phase_updates["eval_limit"] = int(args.eval_limit)
    if phase_updates:
        overrides["phase4"] = phase_updates
    return load_structured_config(args.config, overrides)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_phase4(config_from_args(args), max_runs=args.max_runs, max_modules=args.max_modules)


if __name__ == "__main__":
    main()
