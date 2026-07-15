from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fit_powerlaw import bootstrap_powerlaw_ci
from .hooks import ActivationHookManager, HookKey, resolve_layers, split_qkv
from .kv_cache_analysis import iter_kv_cache_arrays
from .spectral_metrics import covariance_eigenspectrum, summarize_spectrum
from .streaming_covariance import RunningCovariance, RunningMoments, sample_space_covariance_eigenvalues
from .time_lag_dynamics import dynamic_summary_rows


SAMPLE_TEXTS = [
    "Language models transform a sequence of symbols into layered vector states. "
    "Those states can be studied as samples from a high dimensional dynamical system.",
    "In-context learning, induction patterns, and retrieval behavior all leave traces in residual streams, attention outputs, and feed-forward activations.",
    "Spectral analysis asks whether variance is spread across many directions or concentrated in a small number of dominant modes.",
    "The quick brown fox writes a proof, checks a matrix factorization, and compares the trained network to a random initialization.",
]

STRUCTURED_PROMPTS = [
    "A B C D A B C D A B C D",
    "1 + 1 = 2\n2 + 2 = 4\n3 + 3 = 6\n4 + 4 =",
    "Question: Paris is in which country?\nAnswer: France\nQuestion: Rome is in which country?\nAnswer:",
    "copy: red blue green red blue green red blue",
]

SUPPORTED_SITES = {
    "resid_pre",
    "resid_mid",
    "resid_post",
    "attn_out",
    "q",
    "k",
    "v",
    "pattern",
    "mlp_out",
    "k_cache",
    "v_cache",
}

PLANNED_UNSUPPORTED_SITES = {
    "ffn_intermediate",
    "attention_logits",
    "attention_probabilities",
}


@dataclass
class CollectionResult:
    metric_rows: list[dict[str, object]]
    dynamic_rows: list[dict[str, object]]
    eigen_payloads: list[dict[str, object]]
    metadata: dict[str, object]


def require_torch_transformers():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and HuggingFace Transformers are required for real model collection. "
            "Install project dependencies or run with --synthetic-smoke."
        ) from exc
    return torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate_sites(sites: list[str]) -> None:
    unsupported = sorted(set(sites) - SUPPORTED_SITES)
    if unsupported:
        planned = sorted(set(unsupported) & PLANNED_UNSUPPORTED_SITES)
        unknown = sorted(set(unsupported) - PLANNED_UNSUPPORTED_SITES)
        details: list[str] = []
        if planned:
            details.append(f"planned but not implemented yet: {planned}")
        if unknown:
            details.append(f"unknown: {unknown}")
        raise ValueError(
            "unsupported activation site(s): "
            + "; ".join(details)
            + f". Supported sites are: {sorted(SUPPORTED_SITES)}"
        )


def hook_sites_for_requested_sites(sites: list[str]) -> list[str]:
    hooks: set[str] = set()
    for site in sites:
        if site in {"resid_post", "attn_out", "mlp_out"}:
            hooks.add(site)
        elif site in {"q", "k", "v"}:
            hooks.add("qkv")
        elif site == "resid_mid":
            hooks.add("attn_out")
    return sorted(hooks)


def torch_dtype_from_name(torch: Any, dtype_name: str | None):
    if dtype_name is None or str(dtype_name).strip() == "":
        return None
    name = str(dtype_name).strip().lower()
    if name == "auto":
        return "auto"
    aliases = {
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    attr = aliases.get(name, name)
    if not hasattr(torch, attr):
        raise ValueError(f"unsupported torch dtype: {dtype_name}")
    return getattr(torch, attr)


def model_input_device(model: Any):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        try:
            return next(model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("model has no parameters; cannot infer input device") from exc


def load_model_and_tokenizer(
    model_name: str,
    *,
    variant: str = "pretrained",
    revision: str | None = None,
    device: str = "auto",
    trust_remote_code: bool = True,
    device_map: str | None = None,
    torch_dtype: str | None = None,
    local_files_only: bool = False,
    low_cpu_mem_usage: bool = False,
):
    torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer = require_torch_transformers()
    dtype = torch_dtype_from_name(torch, torch_dtype)
    from_pretrained_kwargs = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "local_files_only": bool(local_files_only),
    }
    config = AutoConfig.from_pretrained(model_name, **from_pretrained_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **from_pretrained_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if variant == "pretrained":
        model_kwargs = dict(from_pretrained_kwargs)
        model_kwargs["low_cpu_mem_usage"] = bool(low_cpu_mem_usage)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if device_map and str(device_map).lower() not in {"none", "false", "0"}:
            model_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
    elif variant == "random":
        if dtype not in {None, "auto"}:
            config.torch_dtype = dtype
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    else:
        raise ValueError(f"unsupported variant: {variant}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_device_map = hasattr(model, "hf_device_map") and bool(getattr(model, "hf_device_map"))
    if has_device_map:
        device = str(model_input_device(model))
    else:
        model.to(device)
    model.eval()
    return model, tokenizer, device


def make_input_ids(
    tokenizer: Any,
    *,
    condition: str,
    num_sequences: int,
    seq_len: int,
    seed: int,
    vocab_size: int | None = None,
):
    torch, *_ = require_torch_transformers()
    rng = np.random.default_rng(seed)
    if condition == "random_uniform":
        high = int(vocab_size or tokenizer.vocab_size)
        arr = rng.integers(0, high, size=(num_sequences, seq_len), dtype=np.int64)
        return torch.tensor(arr, dtype=torch.long)

    if condition == "structured":
        text = "\n".join(STRUCTURED_PROMPTS)
    else:
        text = "\n\n".join(SAMPLE_TEXTS)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.eos_token_id]
    needed = num_sequences * seq_len
    reps = int(np.ceil((needed + 1) / len(token_ids)))
    long_ids = (token_ids * reps)[:needed]
    arr = np.asarray(long_ids, dtype=np.int64).reshape(num_sequences, seq_len)
    return torch.tensor(arr, dtype=torch.long)


def batch_iter(input_ids: Any, batch_size: int):
    for start in range(0, int(input_ids.shape[0]), int(batch_size)):
        yield input_ids[start : start + batch_size]


def activation_to_matrix(tensor: Any, *, exclude_first_tokens: int = 0) -> np.ndarray:
    if tensor is None:
        raise ValueError("missing activation tensor")
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"activation must have at least two dimensions, got {arr.shape}")
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    if exclude_first_tokens > 0:
        if arr.shape[1] <= exclude_first_tokens:
            return np.empty((0, arr.shape[-1]), dtype=np.float64)
        arr = arr[:, exclude_first_tokens:]
    return arr.reshape(-1, arr.shape[-1]).astype(np.float64, copy=False)


def activation_to_sequences(tensor: Any, *, exclude_first_tokens: int = 0) -> np.ndarray:
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected [batch, time, dim] activation, got {arr.shape}")
    if exclude_first_tokens > 0:
        if arr.shape[1] <= exclude_first_tokens:
            return np.empty((arr.shape[0], 0, arr.shape[-1]), dtype=np.float64)
        arr = arr[:, exclude_first_tokens:]
    return arr.astype(np.float64, copy=False)


def _token_nll(outputs: Any, input_ids: Any):
    import torch

    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return nll.detach().cpu().float().numpy()


def _eigenspectrum_from_accumulator(acc: RunningCovariance, *, output_zscore: bool) -> tuple[np.ndarray, str, np.ndarray]:
    samples = acc.sample_array()
    dim = int(acc.dim or (samples.shape[1] if samples.ndim == 2 else 0))
    if samples.shape[0] >= 2 and dim > 0 and samples.shape[0] < dim:
        values = samples
        if output_zscore:
            centered = values - values.mean(axis=0, keepdims=True)
            std = values.std(axis=0, ddof=1, keepdims=True)
            values = np.divide(centered, std, out=np.zeros_like(centered), where=std > 1e-12)
        eig = sample_space_covariance_eigenvalues(values, center=not output_zscore)
        return eig, "sample_space_reservoir", samples
    cov = acc.correlation() if output_zscore else acc.covariance()
    return covariance_eigenspectrum(cov), "covariance_eigh", samples


def _site_tensor(outputs: Any, hook_mgr: ActivationHookManager, layer: int, site: str):
    hidden_states = getattr(outputs, "hidden_states", None)
    if site == "resid_pre":
        if hidden_states is None:
            return None
        return hidden_states[layer]
    if site == "resid_post":
        tensor = hook_mgr.cache.get(HookKey(layer, "resid_post"))
        if tensor is not None:
            return tensor
        if hidden_states is None:
            return None
        return hidden_states[layer + 1]
    if site == "resid_mid":
        if hidden_states is None:
            return None
        attn = hook_mgr.cache.get(HookKey(layer, "attn_out"))
        if attn is None:
            return None
        return hidden_states[layer] + attn
    if site in {"attn_out", "mlp_out"}:
        return hook_mgr.cache.get(HookKey(layer, site))
    if site in {"q", "k", "v"}:
        qkv = hook_mgr.cache.get(HookKey(layer, "qkv"))
        return None if qkv is None else split_qkv(qkv, site)
    if site == "pattern":
        attentions = getattr(outputs, "attentions", None)
        if attentions is None or layer >= len(attentions) or attentions[layer] is None:
            return None
        # [batch, heads, query, key] -> one row per query position.
        return attentions[layer].transpose(1, 2).reshape(attentions[layer].shape[0], attentions[layer].shape[2], -1)
    return None


def collect_activation_statistics(
    *,
    model_name: str,
    variant: str,
    sites: list[str],
    layers: str | list[int],
    num_sequences: int,
    seq_len: int,
    batch_size: int,
    dataset_condition: str,
    seed: int,
    output_zscore: bool,
    exclude_first_tokens: int,
    sample_limit: int,
    powerlaw_rank_min: int,
    powerlaw_rank_max: int | None,
    bootstrap_samples: int,
    dynamic_enabled: bool,
    dynamic_site: str,
    dynamic_layer: int | str,
    dynamic_pca_rank: int,
    dynamic_max_sequences: int,
    dynamic_lags: list[int],
    revision: str | None = None,
    device: str = "auto",
    device_map: str | None = None,
    torch_dtype: str | None = None,
    local_files_only: bool = False,
    low_cpu_mem_usage: bool = False,
) -> CollectionResult:
    validate_sites(sites)
    set_seed(seed)
    torch, *_ = require_torch_transformers()
    model, tokenizer, device = load_model_and_tokenizer(
        model_name,
        variant=variant,
        revision=revision,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    selected_layers = resolve_layers(model, layers)
    n_layers = max(selected_layers) + 1 if selected_layers else 0
    if dynamic_layer == "last":
        dynamic_layer_idx = selected_layers[-1]
    else:
        dynamic_layer_idx = int(dynamic_layer)
        if dynamic_layer_idx < 0:
            dynamic_layer_idx = n_layers + dynamic_layer_idx

    input_ids = make_input_ids(
        tokenizer,
        condition=dataset_condition,
        num_sequences=num_sequences,
        seq_len=seq_len,
        seed=seed,
        vocab_size=getattr(model.config, "vocab_size", None),
    )
    input_ids = input_ids.to(device)

    hook_mgr = ActivationHookManager(model, selected_layers, hook_sites_for_requested_sites(sites))
    hook_mgr.install()
    accumulators: dict[tuple[int, str], RunningCovariance] = {}
    kv_accumulators: dict[tuple[int, int, str], RunningCovariance] = {}
    nll_stats = RunningMoments()
    dynamic_sequences: list[np.ndarray] = []
    needs_attentions = "pattern" in sites
    needs_cache = any(site in {"k_cache", "v_cache"} for site in sites)
    needs_hidden_states = any(site in {"resid_pre", "resid_mid"} for site in sites)

    try:
        with torch.no_grad():
            for batch in batch_iter(input_ids, batch_size):
                hook_mgr.clear()
                outputs = model(
                    input_ids=batch,
                    output_hidden_states=needs_hidden_states,
                    output_attentions=needs_attentions,
                    use_cache=needs_cache,
                )
                nll_stats.update(_token_nll(outputs, batch).reshape(-1))
                if needs_cache:
                    for layer, head, site, values in iter_kv_cache_arrays(
                        outputs.past_key_values,
                        exclude_first_tokens=exclude_first_tokens,
                    ):
                        if site not in sites or layer not in selected_layers:
                            continue
                        key = (layer, head, site)
                        if key not in kv_accumulators:
                            kv_accumulators[key] = RunningCovariance(sample_limit=sample_limit, seed=seed + layer + head)
                        kv_accumulators[key].update(values)
                for layer in selected_layers:
                    for site in sites:
                        if site in {"k_cache", "v_cache"}:
                            continue
                        tensor = _site_tensor(outputs, hook_mgr, layer, site)
                        if tensor is None:
                            continue
                        matrix = activation_to_matrix(tensor, exclude_first_tokens=exclude_first_tokens)
                        if matrix.shape[0] == 0:
                            continue
                        key = (layer, site)
                        if key not in accumulators:
                            accumulators[key] = RunningCovariance(sample_limit=sample_limit, seed=seed + layer)
                        accumulators[key].update(matrix)
                        if (
                            dynamic_enabled
                            and site == dynamic_site
                            and layer == dynamic_layer_idx
                            and len(dynamic_sequences) < dynamic_max_sequences
                        ):
                            seq_np = activation_to_sequences(tensor, exclude_first_tokens=exclude_first_tokens)
                            if seq_np.shape[1] == 0:
                                continue
                            take = min(seq_np.shape[0], dynamic_max_sequences - len(dynamic_sequences))
                            dynamic_sequences.extend([seq_np[i] for i in range(take)])
    finally:
        hook_mgr.close()

    metric_rows: list[dict[str, object]] = []
    eigen_payloads: list[dict[str, object]] = []
    for (layer, site), acc in sorted(accumulators.items()):
        eig, eigenspectrum_method, samples = _eigenspectrum_from_accumulator(acc, output_zscore=output_zscore)
        fit = bootstrap_powerlaw_ci(
            eig,
            rank_min=powerlaw_rank_min,
            rank_max=powerlaw_rank_max,
            n_boot=bootstrap_samples,
            seed=seed + layer,
        )
        row: dict[str, object] = {
            "model": model_name,
            "variant": variant,
            "revision": revision or "",
            "dataset_condition": dataset_condition,
            "layer": int(layer),
            "site": site,
            "n_samples": int(acc.count),
            "nll_mean": float(nll_stats.mean),
            "nll_std": float(nll_stats.std),
            "zscored": bool(output_zscore),
            "eigenspectrum_method": eigenspectrum_method,
        }
        row.update(summarize_spectrum(eig, powerlaw=fit, samples=samples))
        metric_rows.append(row)
        eigen_payloads.append(
            {
                "model": model_name,
                "variant": variant,
                "revision": revision or "",
                "dataset_condition": dataset_condition,
                "layer": int(layer),
                "site": site,
                "eigenvalues": eig,
                "normalized_eigenvalues": eig / max(float(eig.sum()), 1e-30),
                "eigenspectrum_method": eigenspectrum_method,
                "mean": acc.mean,
                "count": int(acc.count),
            }
        )

    for (layer, head, site), acc in sorted(kv_accumulators.items()):
        eig, eigenspectrum_method, samples = _eigenspectrum_from_accumulator(acc, output_zscore=output_zscore)
        fit = bootstrap_powerlaw_ci(
            eig,
            rank_min=powerlaw_rank_min,
            rank_max=powerlaw_rank_max,
            n_boot=bootstrap_samples,
            seed=seed + layer + head,
        )
        row = {
            "model": model_name,
            "variant": variant,
            "revision": revision or "",
            "dataset_condition": dataset_condition,
            "layer": int(layer),
            "head": int(head),
            "site": site,
            "n_samples": int(acc.count),
            "nll_mean": float(nll_stats.mean),
            "nll_std": float(nll_stats.std),
            "zscored": bool(output_zscore),
            "eigenspectrum_method": eigenspectrum_method,
        }
        row.update(summarize_spectrum(eig, powerlaw=fit, samples=samples))
        metric_rows.append(row)
        eigen_payloads.append(
            {
                "model": model_name,
                "variant": variant,
                "revision": revision or "",
                "dataset_condition": dataset_condition,
                "layer": int(layer),
                "head": int(head),
                "site": site,
                "eigenvalues": eig,
                "normalized_eigenvalues": eig / max(float(eig.sum()), 1e-30),
                "eigenspectrum_method": eigenspectrum_method,
                "mean": acc.mean,
                "count": int(acc.count),
            }
        )

    dynamic_rows: list[dict[str, object]] = []
    if dynamic_enabled and dynamic_sequences:
        lag_rows, dmd_summary = dynamic_summary_rows(
            dynamic_sequences,
            lags=dynamic_lags,
            pca_rank=dynamic_pca_rank,
            dmd_rank=dynamic_pca_rank,
        )
        for row in lag_rows:
            row.update(
                {
                    "model": model_name,
                    "variant": variant,
                    "revision": revision or "",
                    "dataset_condition": dataset_condition,
                    "layer": int(dynamic_layer_idx),
                    "site": dynamic_site,
                    "kind": "lag",
                }
            )
            dynamic_rows.append(row)
        dmd_row = {
            "model": model_name,
            "variant": variant,
            "revision": revision or "",
            "dataset_condition": dataset_condition,
            "layer": int(dynamic_layer_idx),
            "site": dynamic_site,
            "kind": "dmd",
            "tau": "",
        }
        dmd_row.update(dmd_summary)
        dynamic_rows.append(dmd_row)

    return CollectionResult(
        metric_rows=metric_rows,
        dynamic_rows=dynamic_rows,
        eigen_payloads=eigen_payloads,
        metadata={
            "model": model_name,
            "variant": variant,
            "revision": revision or "",
            "device": device,
            "device_map": device_map or "",
            "torch_dtype": torch_dtype or "",
            "local_files_only": bool(local_files_only),
            "low_cpu_mem_usage": bool(low_cpu_mem_usage),
            "num_sequences": int(num_sequences),
            "sequence_length": int(seq_len),
            "batch_size": int(batch_size),
            "sites": sites,
            "layers": selected_layers,
        },
    )


def collect_synthetic_statistics(
    *,
    model_name: str,
    variant: str,
    sites: list[str],
    layers: list[int],
    num_sequences: int,
    seq_len: int,
    seed: int,
    sample_limit: int,
    powerlaw_rank_min: int,
    powerlaw_rank_max: int | None,
    bootstrap_samples: int,
    dynamic_enabled: bool,
    dynamic_site: str,
    dynamic_layer: int | str,
    dynamic_pca_rank: int,
    dynamic_max_sequences: int,
    dynamic_lags: list[int],
    output_zscore: bool = False,
    exclude_first_tokens: int = 0,
) -> CollectionResult:
    validate_sites(sites)
    rng = np.random.default_rng(seed)
    dim = 32
    metric_rows: list[dict[str, object]] = []
    eigen_payloads: list[dict[str, object]] = []
    dynamic_sequences: list[np.ndarray] = []
    dyn_layer = layers[-1] if dynamic_layer == "last" else int(dynamic_layer)
    site_scale = {"resid_post": 1.0, "attn_out": 0.7, "mlp_out": 1.4}
    variant_scale = 1.0 if variant == "pretrained" else 0.85

    for layer in layers:
        for site in sites:
            alpha = 1.2 + 0.08 * layer + (0.15 if variant == "pretrained" else 0.0)
            spectrum = (np.arange(1, dim + 1, dtype=np.float64) ** (-alpha)) * site_scale.get(site, 1.0) * variant_scale
            basis, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
            raw_samples = rng.normal(size=(num_sequences * seq_len, dim)) @ (basis * np.sqrt(spectrum))
            seqs = raw_samples.reshape(num_sequences, seq_len, dim)
            if exclude_first_tokens > 0:
                if seq_len <= exclude_first_tokens:
                    analysis_seqs = np.empty((num_sequences, 0, dim), dtype=np.float64)
                else:
                    analysis_seqs = seqs[:, exclude_first_tokens:, :]
            else:
                analysis_seqs = seqs
            samples = analysis_seqs.reshape(-1, dim)
            acc = RunningCovariance(sample_limit=sample_limit, seed=seed + layer)
            if samples.shape[0] == 0:
                continue
            acc.update(samples)
            eig, eigenspectrum_method, samples_for_summary = _eigenspectrum_from_accumulator(acc, output_zscore=output_zscore)
            fit = bootstrap_powerlaw_ci(
                eig,
                rank_min=powerlaw_rank_min,
                rank_max=powerlaw_rank_max,
                n_boot=bootstrap_samples,
                seed=seed + layer,
            )
            row: dict[str, object] = {
                "model": model_name,
                "variant": variant,
                "revision": "",
                "dataset_condition": "synthetic_longtail",
                "layer": int(layer),
                "site": site,
                "n_samples": int(acc.count),
                "nll_mean": float(3.0 - 0.2 * (variant == "pretrained") + 0.01 * layer),
                "nll_std": 0.1,
                "zscored": bool(output_zscore),
                "eigenspectrum_method": eigenspectrum_method,
            }
            row.update(summarize_spectrum(eig, powerlaw=fit, samples=samples_for_summary))
            metric_rows.append(row)
            eigen_payloads.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "revision": "",
                    "dataset_condition": "synthetic_longtail",
                    "layer": int(layer),
                    "site": site,
                    "eigenvalues": eig,
                    "normalized_eigenvalues": eig / max(float(eig.sum()), 1e-30),
                    "eigenspectrum_method": eigenspectrum_method,
                    "mean": acc.mean,
                    "count": int(acc.count),
                }
            )
            if dynamic_enabled and site == dynamic_site and layer == dyn_layer:
                dynamic_sequences.extend([analysis_seqs[i] for i in range(min(num_sequences, dynamic_max_sequences)) if analysis_seqs.shape[1] > 0])

    dynamic_rows: list[dict[str, object]] = []
    if dynamic_enabled and dynamic_sequences:
        lag_rows, dmd_summary = dynamic_summary_rows(
            dynamic_sequences,
            lags=dynamic_lags,
            pca_rank=min(dynamic_pca_rank, dim),
            dmd_rank=min(dynamic_pca_rank, dim),
        )
        for row in lag_rows:
            row.update(
                {
                    "model": model_name,
                    "variant": variant,
                    "revision": "",
                    "dataset_condition": "synthetic_longtail",
                    "layer": int(dyn_layer),
                    "site": dynamic_site,
                    "kind": "lag",
                }
            )
            dynamic_rows.append(row)
        dmd_row = {
            "model": model_name,
            "variant": variant,
            "revision": "",
            "dataset_condition": "synthetic_longtail",
            "layer": int(dyn_layer),
            "site": dynamic_site,
            "kind": "dmd",
            "tau": "",
        }
        dmd_row.update(dmd_summary)
        dynamic_rows.append(dmd_row)

    return CollectionResult(
        metric_rows=metric_rows,
        dynamic_rows=dynamic_rows,
        eigen_payloads=eigen_payloads,
        metadata={
            "model": model_name,
            "variant": variant,
            "synthetic": True,
            "num_sequences": int(num_sequences),
            "sequence_length": int(seq_len),
            "sites": sites,
            "layers": layers,
        },
    )
