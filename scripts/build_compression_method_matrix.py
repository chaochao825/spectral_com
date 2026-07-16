#!/usr/bin/env python3
"""Build the auditable external compression-method comparison matrix.

The matrix deliberately records method scope, optimization signal, mutable state,
and *actual* deployment payload.  It is a literature protocol registry, not a
table of measured results from this repository.
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import Counter
from pathlib import Path
from typing import Iterable


VERIFIED_AS_OF = "2026-07-16"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "compression_method_comparison_20260713"
)

METHOD_COLUMNS = [
    "method_id",
    "method",
    "variant",
    "lane",
    "sub_lane",
    "lane_definition",
    "scope",
    "gradient_training_signal",
    "updated_state",
    "objective",
    "payload_must_count",
    "strict_ptq_direct_comparison",
    "comparison_conditions",
    "reproduction_status",
    "primary_paper",
    "official_repo",
    "evidence_note",
    "verified_as_of",
]

LANE_DEFINITIONS = {
    "A0": "A：冻结稠密模型、无反向传播的 PTQ；A0：data-free，不使用校准样本或激活统计。",
    "A1": "A：冻结稠密模型、无反向传播的 PTQ；A1：使用校准样本、激活统计或闭式二阶量，但不做梯度更新。",
    "B": "B：冻结 dense pretrained base 的校准型 PTQ；允许为局部重构、曲率统计执行 backward/HVP/STE，或优化最终可融合的量化/辅助状态，但不做全局任务恢复。",
    "C": "C：使用全局 CE/KL 梯度优化、QAT、下游 PEFT/微调，或主结果流水线以任务损失恢复全模型行为。",
    "D": "D：非同范围；KV-cache、仅激活、原生低比特训练、NAS/结构搜索等不能作为严格权重 PTQ 对照。",
}

ALLOWED_COMPARISON = {
    "yes_if_protocol_matched",
    "mechanism_only",
    "no_scope_mismatch",
}
URL_SENTINELS = {
    "not_applicable_baseline",
    "not_found_in_primary_sources",
}


def entry(
    method_id: str,
    method: str,
    variant: str,
    lane: str,
    sub_lane: str,
    scope: str,
    gradient_training_signal: str,
    updated_state: str,
    objective: str,
    payload_must_count: str,
    strict_ptq_direct_comparison: str,
    comparison_conditions: str,
    reproduction_status: str,
    primary_paper: str,
    official_repo: str,
    evidence_note: str,
) -> dict[str, str]:
    """Create one normalized matrix row."""

    return {
        "method_id": method_id,
        "method": method,
        "variant": variant,
        "lane": lane,
        "sub_lane": sub_lane,
        "lane_definition": LANE_DEFINITIONS[sub_lane],
        "scope": scope,
        "gradient_training_signal": gradient_training_signal,
        "updated_state": updated_state,
        "objective": objective,
        "payload_must_count": payload_must_count,
        "strict_ptq_direct_comparison": strict_ptq_direct_comparison,
        "comparison_conditions": comparison_conditions,
        "reproduction_status": reproduction_status,
        "primary_paper": primary_paper,
        "official_repo": official_repo,
        "evidence_note": evidence_note,
        "verified_as_of": VERIFIED_AS_OF,
    }


def build_rows() -> list[dict[str, str]]:
    """Return the hand-audited method registry.

    Numeric accuracy, speed, and ranking claims are intentionally excluded.
    Protocol numbers are retained only where primary sources make them necessary
    to disambiguate a variant or to account for its deployment payload.
    """

    rows = [
        entry(
            "rtn", "RTN", "round-to-nearest baseline", "A", "A0",
            "weight-only uniform quantization baseline",
            "none; deterministic rounding from weight ranges",
            "quantized weight codes and scale/zero-point only",
            "local rounding error without calibration objective",
            "packed weight codes; scale/zero-point; group metadata; padding; file/header/alignment",
            "yes_if_protocol_matched",
            "same model, tensors, group size, clipping rule, excluded tensors, actual packed bytes, and evaluation set",
            "native_baseline_implemented; measured_in_existing_exploratory_runs",
            "not_applicable_baseline", "not_applicable_baseline",
            "Baseline definition; no single canonical paper or official repository is asserted.",
        ),
        entry(
            "gptq", "GPTQ", "one-shot second-order weight quantization", "A", "A1",
            "post-training weight-only quantization",
            "no backpropagation; calibration activations form an approximate Hessian",
            "quantized codes while processing columns; frozen dense model",
            "approximately minimize layer reconstruction error under second-order weighting",
            "packed codes; scale/zero-point; grouping/permutation metadata; unquantized tensors; padding/header/alignment",
            "yes_if_protocol_matched",
            "match model, calibration data/tokens, bit/group configuration, tensor coverage, actual bytes, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2210.17323", "https://github.com/IST-DASLab/gptq",
            "Scope and optimization characterization follow the paper and official implementation.",
        ),
        entry(
            "awq", "AWQ", "activation-aware weight quantization", "A", "A1",
            "post-training weight-only quantization",
            "no backpropagation; calibration activation statistics guide salient-weight protection and scaling",
            "quantized codes and equivalent/foldable channel scales; dense weights otherwise frozen",
            "reduce quantization error by activation-aware rescaling and clipping/search",
            "packed codes; scale/zero-point; any non-folded channel scales; tensor exclusions; padding/header/alignment",
            "yes_if_protocol_matched",
            "match calibration, group size, clipping/search, protected tensors, folded-state verification, actual bytes, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2306.00978", "https://github.com/mit-han-lab/llm-awq",
            "Paper and official repository support a calibration-statistics, no-gradient PTQ classification.",
        ),
        entry(
            "sparsegpt", "SparseGPT", "one-shot unstructured or semi-structured pruning", "A", "A1",
            "post-training weight pruning",
            "no backpropagation; calibration activations form a Hessian approximation",
            "sparse weight values and support; dense model frozen",
            "second-order layer reconstruction while removing weights",
            "surviving values; mask/index or N:M metadata; block metadata; padding/header/alignment; any dense exceptions",
            "yes_if_protocol_matched",
            "match sparsity structure, calibration, model/tensor coverage, deployable sparse encoding bytes, kernels, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2301.00774", "https://github.com/IST-DASLab/sparsegpt",
            "The paper and official code establish the one-shot Hessian-informed pruning scope.",
        ),
        entry(
            "wanda", "Wanda", "activation-magnitude pruning", "A", "A1",
            "post-training weight pruning",
            "no backpropagation; calibration activation magnitudes score weights",
            "sparse values and support; dense model frozen",
            "retain weights with large weight-times-activation importance",
            "surviving values; mask/index or N:M metadata; padding/header/alignment; dense exceptions",
            "yes_if_protocol_matched",
            "match sparsity pattern, calibration data, tensor coverage, actual sparse artifact bytes, kernels, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2306.11695", "https://github.com/locuslab/wanda",
            "Classification follows the paper's pruning metric and official implementation.",
        ),
        entry(
            "spqr", "SpQR", "sparse-quantized representation", "A", "A1",
            "weight-only quantization with sparse outliers",
            "no model backpropagation; calibration-based second-order sensitivity separates outliers",
            "quantized bulk weights plus sparse high-precision outliers",
            "minimize reconstruction loss while preserving sensitive outliers",
            "bulk codes/scales/zero-points; outlier values and indices; permutations; padding/header/alignment",
            "yes_if_protocol_matched",
            "match outlier policy, calibration, exact tensor coverage, serialized sparse/index overhead, kernels, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2306.03078", "https://github.com/Vahe1994/SpQR",
            "Hybrid payload and second-order outlier treatment are documented by the primary sources.",
        ),
        entry(
            "squeezellm", "SqueezeLLM", "dense-and-sparse non-uniform quantization", "B", "B",
            "weight-only PTQ with non-uniform codebooks and sparse outliers",
            "no full-model parameter training, but the main sensitivity path performs backward in a separate framework to save squared loss gradients as a Fisher-based sensitivity checkpoint",
            "non-uniform weight indices/codebooks plus sparse components",
            "sensitivity-weighted quantization error with outlier separation",
            "indices; codebooks; scales; sparse outlier values/indices; tensor exceptions; padding/header/alignment",
            "yes_if_protocol_matched",
            "match the gradient corpus/objective and backward budget, codebook/outlier configuration, artifact bytes, runtime format, and evaluation; do not compare this main Fisher path as no-backward A1 PTQ",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2306.07629", "https://github.com/SqueezeAILab/SqueezeLLM",
            "The official quantization README requires a gradient-square checkpoint before sensitivity-weighted k-means; encoder-side gradients are not deployed, but their backward computation places the main path in B.",
        ),
        entry(
            "lqer", "LQER", "low-rank quantization error reconstruction", "A", "A1",
            "weight-only PTQ plus low-rank residual repair",
            "no gradient training; activation-aware statistics can shape a low-rank error decomposition",
            "base quantized weights and low-rank correction factors",
            "approximate the quantization residual in a low-rank subspace relevant to activations",
            "base codes/scales/zero-points; both low-rank factors with dtype/scales; rank metadata; padding/header/alignment",
            "yes_if_protocol_matched",
            "match rank, factor dtype, calibration, base codec, actual repair bytes, tensor coverage, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2402.02446", "https://github.com/ChengZhang-98/lqer",
            "Low-rank repair state must be charged in addition to the base codec.",
        ),
        entry(
            "qera", "QERA", "quantization error reconstruction analysis", "A", "A1",
            "weight-only PTQ plus low-rank error reconstruction",
            "closed-form/statistical reconstruction without end-to-end weight fine-tuning",
            "quantized base and low-rank reconstruction factors",
            "minimize activation-weighted output reconstruction error",
            "base codec; low-rank factor values/dtypes/scales; rank and grouping metadata; padding/header/alignment",
            "yes_if_protocol_matched",
            "match calibration covariance, rank/dtype, base quantizer, serialized bytes, and evaluation protocol",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2410.06040", "https://github.com/ChengZhang-98/QERA",
            "Method scope and reconstruction objective follow the primary paper and repository.",
        ),
        entry(
            "eora", "EoRA", "eigen-oriented low-rank adaptation", "A", "A1",
            "weight-only PTQ with calibration-oriented low-rank compensation",
            "no full-model backpropagation in the PTQ repair stage; calibration eigenspace guides repair",
            "quantized base and low-rank compensation factors",
            "reconstruct quantization error in activation-sensitive eigen-directions",
            "base codec; low-rank factors and their dtype/scales; eigenspace/rank metadata if stored; padding/header/alignment",
            "yes_if_protocol_matched",
            "match calibration, rank, factor precision, base codec, actual bytes, tensor coverage, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2410.21271", "https://github.com/NVlabs/EoRA",
            "The official paper/repository motivate an activation-sensitive low-rank repair comparison.",
        ),
        entry(
            "quip_sharp", "QuIP#", "incoherence processing plus vector quantization", "C", "C",
            "weight-only compression; main reported pipeline includes fine-tuning",
            "reported pipeline uses optimization/fine-tuning after quantization; not a frozen closed-form-only PTQ control",
            "lattice-quantized weights and pipeline-tuned state",
            "second-order proxy plus end-to-end recovery in the reported pipeline",
            "lattice bitstream; scales; RHT signs/seeds; codebooks/decoder state; tuned or unquantized tensors; padding/header/alignment",
            "mechanism_only",
            "can compare deployment frontier only after matching actual bytes and disclosing fine-tuning data/steps/compute; not strict frozen-PTQ evidence",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2402.04396", "https://github.com/Cornell-RelaxML/quip-sharp",
            "Placed in C because the main official pipeline includes fine-tuning; codec components remain informative.",
        ),
        entry(
            "qtip", "QTIP", "trellis-coded quantization", "A", "A1",
            "weight-only PTQ with incoherence processing and trellis coding",
            "no end-to-end fine-tuning asserted for the core PTQ path; calibration/second-order information guides quantization",
            "trellis-coded weights and decode state",
            "minimize second-order weighted quantization distortion under a compressed trellis representation",
            "trellis streams/state/decoder parameters; scales; RHT signs/seeds; tensor exceptions; padding/header/alignment",
            "yes_if_protocol_matched",
            "match code rate, decoder state, calibration, transform representation, serialized bytes, kernels, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2406.11235", "https://github.com/Cornell-RelaxML/qtip",
            "Trellis decoder and transform state are part of deployable payload, not free metadata.",
        ),
        entry(
            "aqlm", "AQLM", "additive quantization", "C", "C",
            "weight-only additive-codebook compression; main reported pipeline includes blockwise and end-to-end optimization",
            "gradient optimization and end-to-end KL-distillation are used in the main pipeline",
            "additive code indices, codebooks, scales, and optimized quantization state",
            "block reconstruction followed by global output-distribution recovery",
            "additive-code indices; all codebooks; scales; partition/scheme metadata; unquantized tensors; padding/header/alignment",
            "mechanism_only",
            "deployment frontier can be compared at actual bytes, but training data/steps/compute must be disclosed and it is not frozen PTQ",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2401.06118", "https://github.com/Vahe1994/AQLM",
            "Classified by the main reported training pipeline rather than by nominal weight bit rate alone.",
        ),
        entry(
            "omniquant", "OmniQuant", "learnable weight/activation PTQ", "B", "B",
            "calibration-optimized PTQ for weights and optionally activations",
            "calibration backpropagation optimizes clipping and learnable equivalent transformations; dense pretrained weights remain frozen",
            "quantizer clipping/scale state and transform parameters, folded when supported",
            "block/layer output reconstruction on calibration samples",
            "weight codes/scales/zero-points; activation quantizer state if in scope; any non-folded LET parameters; padding/header/alignment",
            "yes_if_protocol_matched",
            "match W-only versus W/A scope, calibration tokens, gradient steps, updated variables, compute, actual exported bytes, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2308.13137", "https://github.com/OpenGVLab/OmniQuant",
            "The learned PTQ parameters require lane B even though the pretrained dense weights are frozen.",
        ),
        entry(
            "spinquant", "SpinQuant", "learned rotations for quantization", "C", "C",
            "rotation-assisted PTQ for weights and activations/KV depending protocol",
            "pretrained base weights are frozen, but the official optimized-rotation path trains global R1 and per-layer R2 with Hugging Face Trainer on WikiText2 causal-LM loss",
            "global and per-layer rotation parameters are learned under full-model causal-LM behavior; rotations may be fused only when export proves equivalence",
            "global causal language-model loss under simulated low-bit quantization",
            "codes/scales/zero-points; stored rotation matrices or signs/seeds/factors; unfused runtime transforms; padding/header/alignment",
            "mechanism_only",
            "treat optimized SpinQuant as a global-recovery track: match W/A/KV scope, corpus/tokens/epochs/optimizer, rotation export/folding, actual artifact bytes, kernels, and evaluation; do not rank it as local-reconstruction B PTQ",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2405.16406", "https://github.com/facebookresearch/SpinQuant",
            "Official optimize_rotation.py freezes the dense base but calls Trainer.train() on causal-LM data; the global task objective places this optimized path in C. Learned rotations are not free unless demonstrably fused or regenerated from charged metadata.",
        ),
        entry(
            "sliderquant", "SliderQuant", "default; fused channel scales and rank-4 LoRA", "B", "B",
            "calibration-optimized W-only or W/A PTQ with inter-layer windows and intra-layer incremental quantization",
            "AdamW optimizes local calibration reconstruction on 128 samples of length 2048; 20 epochs by default and 60 for W2A16; pretrained dense base stays frozen",
            "learnable channel scales and rank-4 LoRA on all linear layers; the default export absorbs both into adjacent/original weights",
            "sliding-window output reconstruction under simulated low-bit quantization",
            "exported codes/scales/zero-points; channel-scale or LoRA state only if export fails to merge it; scheme/group IDs; uncompressed exceptions; padding/header/alignment",
            "yes_if_protocol_matched",
            "match W-only versus W/A scope, calibration corpus and 128x2048-token budget, 20/60 epochs, optimizer, window schedule, group size, merged-state proof, actual bytes, compute, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2603.25284", "https://github.com/deep-optimization/SliderQuant",
            "B denotes local calibration reconstruction with a frozen dense base; rank-4 LoRA here is auxiliary calibration state, not downstream PEFT.",
        ),
        entry(
            "sliderquant_plus", "SliderQuant", "SliderQuant+ with runtime rotations", "B", "B",
            "SliderQuant calibration path augmented with non-absorbable Hadamard transformations for W/A quantization",
            "uses the same frozen-base AdamW calibration path and local sliding-window reconstruction as SliderQuant; rotation execution is an added deployment mechanism",
            "fused channel scales and rank-4 LoRA plus a fixed runtime Hadamard transform plan",
            "sliding-window output reconstruction under simulated low-bit quantization with rotation transformations",
            "exported codes/scales/zero-points; any non-merged auxiliary state; transform locations and signs/seeds/factors; non-absorbable Hadamard runtime operations, kernels and workspace; scheme/group IDs; padding/header/alignment",
            "yes_if_protocol_matched",
            "match W/A scope, calibration and epoch budget, window/rotation protocol, actual payload, transform runtime and workspace, kernels, latency, and evaluation; do not mix with zero-extra-cost SliderQuant",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2603.25284", "https://github.com/deep-optimization/SliderQuant",
            "Separated because SliderQuant+ adds non-absorbable Hadamard transformations and explicit inference-time cost.",
        ),
        entry(
            "liftquant_block", "LiftQuant", "block correction only", "B", "B",
            "lifted low-bit weight compression with blockwise correction",
            "paper/README protocol says STE block correction on 4096 RedPajama samples of length 2048 and 2 epochs; the pinned parser instead exposes nsamples1/nsamples2 and epochs1/epochs2, so the executable two-stage mapping remains unresolved; no global cross-entropy",
            "binary lifted code Wq and fused transform T*=M T^-1; dense pretrained model otherwise frozen",
            "block output-MSE reconstruction",
            "1-bit lifted codes including ceil(IC/d) padding; per-output scales; either fused decoder T* or unfused M plus inverse-whitening factors, following the actual artifact and never double-counting equivalent representations; uncompressed tensors; headers/alignment",
            "mechanism_only",
            "paper targets 2–3-bit regimes; the current project's 4-bit Q/S/L probe is mechanism-only. A strict run must resolve README --nsamples/--epochs versus the two-stage parser, match model/data/rate, and charge stored fused T* OR stored M plus inverse factors; paper reports FP16 transform overhead about 0.008–0.011 bpw at 70B",
            "official_code_audited; official_unpatched_qwen_smoke_failed; compatibility_patched_layer0_smoke_passed; full_external_reproduction_pending",
            "https://arxiv.org/abs/2606.04050", "https://github.com/Heliulu/LiftQuant",
            "Pinned 72b3875 audit: unmodified Qwen2.5 layer-0 smoke fails on missing Catcher.attention_type under Transformers 4.57; a one-line metadata-forwarding compatibility patch completes a bounded 8-window, 1+1 epoch, layer-0 smoke and writes an artifact, but no PPL/task metric or full-model reproduction was run.",
        ),
        entry(
            "liftquant_e2e", "LiftQuant", "optional end-to-end correction", "C", "C",
            "lifted low-bit compression followed by global quantization-parameter optimization",
            "cross-entropy optimization on 4096 samples of length 4096 for 1 epoch after block correction",
            "continuous quantization parameters such as scales/transforms; quantized representation retained",
            "global language-model cross-entropy",
            "1-bit lifted codes and padding; scales; either fused T* or unfused M plus inverse-whitening factors; all other surviving optimized parameters; uncompressed tensors; headers/alignment; never double-count equivalent transform representations",
            "mechanism_only",
            "paper main results use this stage, so report training corpus/tokens/steps/compute and exact payload; pinned 72b3875 e2efinetune.py cannot reach --help because datautils_block is absent, and unknown extra_args would be silently retained; it is not a strict frozen-PTQ comparison",
            "official_code_audited; e2e_entrypoint_blocked_missing_datautils_block; external_reproduction_pending",
            "https://arxiv.org/abs/2606.04050", "https://github.com/Heliulu/LiftQuant",
            "Exactly separated from the block-only row: main paper results use E2E, while the repository calls it optional for deployment. The current pinned entrypoint is import-blocked and has not been executed.",
        ),
        entry(
            "d2quant", "D²Quant", "DSQ plus DAC", "A", "A1",
            "weight-only PTQ with scale/rounding search and activation-mean correction",
            "no backpropagation or dense fine-tuning; DSQ alternates closed-form/statistical scale and rounding updates; DAC uses calibration forwards",
            "quantized weights/scales plus per-layer hidden-size LayerNorm bias correction; a column scale may be absorbed when export verifies it",
            "weight reconstruction for DSQ and post-attention mean-shift cancellation for DAC",
            "base codes/scales/zero-points; DAC LayerNorm bias in its stored dtype for every layer; non-absorbed DSQ/rotation state; padding/header/alignment",
            "yes_if_protocol_matched",
            "match calibration, base codec, corrected layers, actual absorbed state, DAC bias bytes, artifact format, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2602.02546", "https://github.com/XIANGLONGYAN/D2Quant",
            "A useful small-parameter repair control: zero overhead is allowed only for a scale actually fused into an existing stored scale.",
        ),
        entry(
            "admm_q", "ADMM-Q", "joint Hessian-weighted discrete solver", "A", "A1",
            "layer-wise post-training weight quantization; also usable as the weight-quantizer step inside W/A pipelines",
            "no model backpropagation or parameter fine-tuning; calibration activations form H=X^T X and ADMM alternates closed-form continuous updates, quantization projections, optional grid refresh, and pair-swap local search",
            "final hardened quantized codes and fitted scale/zero-point; dense pretrained weights stay frozen and ADMM primal/dual iterates, eigendecomposition, Hessian, and search state are encoder-side only",
            "jointly minimize Hessian-weighted layer reconstruction error under a discrete quantization constraint",
            "packed codes; scale/zero-point; group/channel and clipping metadata; any separately deployed scaling/rotation state from the surrounding pipeline; unquantized tensors; padding/header/alignment; do not charge encoder-only ADMM/Hessian state unless an exporter serializes it",
            "yes_if_protocol_matched",
            "match calibration tokens, Hessian damping, iteration/penalty/grid-refresh/local-search protocol, clipping and group/channel codec, surrounding transforms, actual serialized bytes, quantization time/memory, and held-out evaluation",
            "no_official_repo_found_as_of_2026-07-14; literature_only; external_reproduction_pending",
            "https://arxiv.org/abs/2605.11222", "not_found_in_primary_sources",
            "ADMM is an iterative encoder-side discrete optimizer, not gradient training. The paper contains no official-code link and no author-designated repository was confirmed by the cutoff date.",
        ),
        entry(
            "has_vq", "HAS-VQ", "Hessian-masked sparse vector quantization", "A", "A1",
            "weight-only PTQ with a vector-quantized dense body and sparse high-sensitivity residual feedback",
            "no backpropagation or trainable model update in the official implementation; 128 length-1024 WikiText-2 calibration forwards estimate diagonal activation second moments, followed by iterative k-means on masked weight blocks",
            "vector index stream and codebooks for the dense body, per-channel scales, plus sparse residual values and their support; dense pretrained model otherwise frozen",
            "minimize diagonal-Hessian-weighted distortion while reconstructing selected high-sensitivity coordinates exactly up to the stored residual precision",
            "packed vector indices including block padding; every codebook centroid and dtype/dimension; channel scales; sparse residual values plus indices/bitmap/row pointers and their dtypes; sparsity/block/scheme metadata; unquantized tensors and biases; padding/header/alignment",
            "yes_if_protocol_matched",
            "match model, calibration samples/tokens, diagonal statistic, block size, centroid count, k-means iterations/sampling, sparsity ratio, sparse codec, actual serialized bytes, kernel/decode path, and held-out evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2601.06959", "https://github.com/VladimerKhasia/HASVQ",
            "The paper's BPP includes indices, codebooks, scales, and sparse residuals; strict exact-rate comparison still requires a real packed artifact rather than the analytical count used by the reference script.",
        ),
        entry(
            "septq", "SEPTQ", "static-global selective GPTQ-style PTQ", "A", "A1",
            "weight-only layer-wise PTQ that leaves globally selected important weights unquantized and quantizes the remaining locations",
            "no gradient training or STE; 128 random length-2048 C4 calibration segments form a Hessian/Cholesky proxy, then a static global mask guides closed-form column-wise error-compensation updates",
            "quantized codes at selected locations and updated higher-precision reserved weights at the complementary support; dense pretrained model is not fine-tuned",
            "minimize constrained second-order layer reconstruction error while protecting a static global set of high-importance weights",
            "packed quantized codes/scales/zero-points; every reserved higher-precision value and its support mask/indices unless a deployed mixed-format codec encodes support implicitly; ratio/block metadata; unquantized tensors; padding/header/alignment",
            "yes_if_protocol_matched",
            "match calibration, Hessian damping/Cholesky, importance ratio, static-global mask rule, block size, reserved-value dtype, support codec, actual serialized bytes, and held-out evaluation",
            "no_official_repo_found_as_of_2026-07-14; literature_only; external_reproduction_pending",
            "https://arxiv.org/abs/2604.10091", "not_found_in_primary_sources",
            "The paper states a GPTQ-based implementation but does not designate a public official repository; selective support and higher-precision reserved values must be charged explicitly.",
        ),
        entry(
            "aaac", "AAAC", "two-codebook activation-aware adaptive coding", "A", "A1",
            "4-bit weight-only PTQ with per-layer scalar codebooks",
            "no gradients/Hessian; calibration activation weights enter scalar k-means codebook construction",
            "two codebooks per layer, quantized indices, and a per-group codebook selector",
            "activation-weighted scalar clustering distortion",
            "4-bit indices; normal scales/zero-points; two 16-entry BF16 codebooks per layer (64 bytes); selector bit per group unless safely encoded in an otherwise unused compatible scale sign bit; padding/header/alignment",
            "yes_if_protocol_matched",
            "match 4-bit W-only scope, calibration, grouping, selector encoding/backend semantics, actual artifact bytes, and evaluation",
            "no_official_repo_found_as_of_2026-07-14; literature_only; external_reproduction_pending",
            "https://arxiv.org/abs/2605.08692", "not_found_in_primary_sources",
            "Payload rules are sourced from the paper/OpenReview artifact; no official GitHub repository was identified by the cutoff date.",
        ),
        entry(
            "daq_delta", "DAQ", "Delta-Aware Quantization", "A", "A0",
            "data-free FP8 quantization of post-training weight deltas relative to an available base checkpoint",
            "no calibration activations, Hessian, or backpropagation; coarse-to-fine scale search uses base and post-trained weights",
            "standard FP8 quantized final model/scales; base checkpoint is an encoder-side prerequisite rather than deployment payload when already shared",
            "maximize delta sign-preservation rate and cosine similarity",
            "FP8 codes; block/per-channel scales; format metadata; padding/header/alignment; charge base checkpoint if the deployment scenario does not already possess it",
            "mechanism_only",
            "narrow boundary: requires paired base/post-trained checkpoints and targets FP8 delta preservation, not generic sub-4-bit PTQ; sign/cosine are secondary diagnostics only",
            "official_documentation_available; external_reproduction_pending",
            "https://arxiv.org/abs/2603.22324", "https://github.com/Tencent/AngelSlim/blob/main/docs/source/features/quantization/daq.md",
            "This is the 2026 Delta-Aware method, not the earlier Density-Aware DAQ; scope is intentionally narrow.",
        ),
        entry(
            "q_palette", "Q-Palette", "data-free palette codec", "A", "A0",
            "weight-only data-free selection among non-uniform, vector, and trellis-style quantization palettes",
            "no calibration dataset or model backpropagation for the core data-free codec/search path",
            "per-layer or partition quantized streams and selected codec scheme",
            "rate-distortion/information-theoretic codec selection",
            "index or trellis streams; scales; required codebooks/LUTs; layer/partition scheme IDs; incoherence-transform signs/seeds/factors when enabled; fusion/merge plan when required; padding/header/alignment",
            "yes_if_protocol_matched",
            "compare actual serialized bytes, not only information-theoretic bits; match model, tensor scope, codec search, kernels, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2509.20214", "https://github.com/snu-mllab/Q-Palette",
            "Closest exact-rate codec control in this registry; theoretical bit objectives do not substitute for packed artifact size.",
        ),
        entry(
            "q_palette_data_aware", "Q-Palette", "data-aware Hessian/perplexity-guided palette codec", "A", "A1",
            "weight-only PTQ selecting mixed quantization schemes under memory or latency constraints",
            "no retraining or trainable-parameter update; QTIP-style proxy Hessian and calibration/validation forward losses guide quantization and scheme selection",
            "per-layer or partition quantized streams, selected codec scheme, and fusion plan",
            "proxy-Hessian quantization distortion plus measured validation perplexity degradation for mixed-scheme allocation",
            "index or trellis streams; scales; required codebooks/LUTs; layer/partition scheme IDs; incoherence-transform signs/seeds/factors; fusion/merge plan; padding/header/alignment",
            "yes_if_protocol_matched",
            "match proxy Hessian, calibration and scheme-selection data, hold selection data separate from final evaluation, memory/latency constraint, codec kernels, actual serialized bytes, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2509.20214", "https://github.com/snu-mllab/Q-Palette",
            "Separated from the data-free row because the paper and official artifacts report a distinct data-aware protocol without retraining.",
        ),
        entry(
            "slim", "SLiM", "quantization + 2:4 sparsity + low-rank compensation", "A", "A1",
            "one-shot joint weight quantization, semi-structured pruning, and low-rank repair",
            "no gradient training in the core one-shot path; calibration statistics support pruning/closed-form compensation",
            "quantized sparse base plus low-rank compensation factors",
            "recover the quantization/pruning residual with a closed-form low-rank component",
            "quantized values/scales; 2:4 mask/metadata; both low-rank factors and dtype/scales; optional adapter if used; padding/header/alignment",
            "yes_if_protocol_matched",
            "use the core no-PEFT path for strict PTQ; match sparsity, rate, rank/dtype, actual sparse/repair bytes, kernels, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2410.09615", "https://github.com/Mohammad-Mozaffari/slim",
            "Optional PEFT must be reported separately in lane C; this row covers only the one-shot core.",
        ),
        entry(
            "obr", "OBR", "closed-form group compensation", "A", "A1",
            "joint quantization/sparsification with block or group residual compensation",
            "closed-form/statistical repair without end-to-end backpropagation",
            "the encoder computes group compensation and folds it into the final sparse low-bit weight before requantization; no separate compensation tensor is assumed at decode",
            "cancel reconstruction residual under the method's groupwise approximation",
            "final quantized/sparse weight codec; masks or N:M metadata; scales/zero-points; any unfused rotation; grouping metadata; padding/header/alignment; charge a compensation tensor only if an exporter actually retains one",
            "yes_if_protocol_matched",
            "match calibration, W/A/KV and sparsity scope, codec, grouping, folded-state proof, actual sparse layout bytes, tensor coverage, kernels and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://openreview.net/forum?id=VQIvBpL5ag", "https://github.com/csguoh/OBR",
            "ICLR 2026 OBR is a closest Hessian joint Q/S comparator. Its encoder-side compensation is folded into the exported sparse low-bit weight unless an inspected artifact proves otherwise.",
        ),
        entry(
            "hestia", "HESTIA", "Hessian-guided differentiable ternary QAT", "C", "C",
            "weight-only group-wise ternary quantization-aware training of pretrained LLM checkpoints",
            "offline Hutch++ Hessian-trace calibration followed by full-model AdamW causal-LM QAT on 10B Ultra-FineWeb tokens; a temperature-controlled softmax relaxation is progressively hardened",
            "latent model weights are globally optimized under the differentiable quantizer; tensor-wise temperature schedules guide training and the final target representation is ternary",
            "global causal language-model loss under Hessian-sensitive soft-to-hard quantization annealing",
            "actual packed ternary codes rather than nominal 1.58 bits; group-128 scales; codebook/quantizer metadata; every excluded full-precision tensor such as non-linear components or lm_head; any state still required by the exported runtime; padding/header/alignment; optimizer, Hessian sketches and temperature schedules are training-only and exempt only if absent from the deployed artifact",
            "mechanism_only",
            "treat as a trained upper-bound track: match initialization, 10B-token corpus and sequence length, global batch/optimizer/schedule, Hutch++ calibration, tensor coverage, actual exported artifact, kernels and evaluation; never compare it as frozen PTQ",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2601.20745", "https://github.com/hestia2026/Hestia",
            "HESTIA belongs to C because the paper performs global causal-LM QAT; offline Hessian calibration does not turn the subsequent 10B-token AdamW optimization into PTQ.",
        ),
        entry(
            "efficientqat", "EfficientQAT", "block-AP plus end-to-end quantization-parameter training", "C", "C",
            "quantization-aware training for low-bit LLM weights and activations depending protocol",
            "blockwise all-parameter training followed by end-to-end quantization-parameter optimization",
            "model/quantizer parameters updated under simulated quantization",
            "block reconstruction and global task/distillation objectives",
            "exported codes/scales/zero-points; remaining quantizer state; any adapters; unquantized tensors; padding/header/alignment",
            "mechanism_only",
            "match W/A scope and actual bytes, but disclose training data/tokens/steps/GPU-hours/peak memory; not strict PTQ",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2407.11062", "https://github.com/OpenGVLab/EfficientQAT",
            "Training scope places the method in C regardless of nominal deployment bit width.",
        ),
        entry(
            "llm_qat", "LLM-QAT", "data-free distillation QAT", "C", "C",
            "QAT covering weights, activations, and optionally KV cache",
            "synthetic/data-free distillation with backpropagation through quantization simulation",
            "student model and quantization parameters",
            "teacher-student language-model distillation under low-bit simulation",
            "all exported W/A/KV quantizer state; packed weights; scales/zero-points; unquantized tensors; padding/header/alignment",
            "mechanism_only",
            "scope must be aligned (W-only versus W/A/KV) and training compute disclosed; not a frozen weight-PTQ control",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2305.17888", "https://github.com/facebookresearch/LLM-QAT",
            "Data-free describes distillation data sourcing, not absence of gradient training.",
        ),
        entry(
            "turboquant", "TurboQuant", "KV-cache/vector quantization", "D", "D",
            "runtime KV-cache compression rather than static model-weight compression",
            "runtime/per-sequence quantization path; training signal is not the current weight-PTQ question",
            "dynamic KV representation and its runtime codec state",
            "reduce KV-cache distortion/memory under vector quantization",
            "KV codes; per-sequence/block scales; vector codebooks; runtime metadata/workspace; kernel requirements",
            "no_scope_mismatch",
            "report separately as a runtime-memory axis; do not merge its rate with static weight bytes",
            "no_official_repo_found_as_of_2026-07-14; literature_only; external_reproduction_pending",
            "https://arxiv.org/abs/2504.19874", "not_found_in_primary_sources",
            "Excluded from strict comparison because the compressed object is KV cache, not model weights.",
        ),
        entry(
            "sharq", "SharQ", "online activation sparse-dense FP4 decomposition with shared FP4 weights", "D", "D",
            "runtime activation decomposition/dispatch plus a shared low-bit weight representation",
            "runtime mechanism; not a like-for-like static weight PTQ optimization",
            "input-adaptive N:M activation mask, sparse backbone, dense residual, and one shared FP4 weight payload with path-specific scale views",
            "reduce activation quantization error and execution cost using sparse/dense paths",
            "runtime activation mask/metadata; shared FP4 weight payload; path-specific scale views; runtime buffers/workspace; fused-preparation and sparse/dense GEMM kernel requirements",
            "no_scope_mismatch",
            "evaluate as activation/runtime system work, not as evidence for a weight-only exact-rate frontier",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2606.26587", "https://github.com/actypedef/SharQ",
            "Scope mismatch is explicit even though the system also consumes low-bit weights.",
        ),
        entry(
            "joint_structural_mixed_precision", "Joint structural pruning + MPQ", "architecture/search method", "D", "D",
            "global structural pruning plus mixed-precision allocation/search",
            "global search/optimization across architecture and bit allocation rather than a fixed-model PTQ transform",
            "selected structure, layer widths/blocks, and mixed-precision weights",
            "global resource-constrained architecture/precision objective",
            "all surviving weights and quantizer state; architecture descriptors; masks/indices; search-selected scheme IDs; padding/header/alignment",
            "no_scope_mismatch",
            "requires a separate architecture-search track; nominal global compression ratio alone is not a strict fixed-model comparison",
            "no_official_repo_found_as_of_2026-07-14; literature_only; external_reproduction_pending",
            "https://arxiv.org/abs/2606.07819", "not_found_in_primary_sources",
            "Kept in lane D because structure/search changes the model definition being compared.",
        ),
        entry(
            "sinq", "SINQ", "calibration-free second-axis scaling", "A", "A0",
            "data-free weight-only PTQ with scaling along an additional axis",
            "no calibration data or gradient training for SINQ core",
            "quantized weights and additional-axis scale state",
            "reduce weight quantization distortion through structured rescaling",
            "codes; ordinary and second-axis scales; zero-points; axis/group metadata; padding/header/alignment",
            "yes_if_protocol_matched",
            "match data-free variant, scale representation, group/axis setup, actual artifact bytes, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2509.22944", "https://github.com/huawei-csl/SINQ",
            "Only the data-free SINQ variant is represented here; extra-axis scales must be charged.",
        ),
        entry(
            "asinq", "A-SINQ", "calibration-assisted second-axis scaling", "A", "A1",
            "calibration-assisted weight-only PTQ",
            "calibration statistics but no end-to-end gradient training in the identified variant",
            "quantized weights and additional-axis scale state",
            "activation-aware quantization distortion reduction",
            "codes; ordinary and second-axis scales; zero-points; axis/group metadata; padding/header/alignment",
            "yes_if_protocol_matched",
            "match calibration, scale representation, actual bytes, group/axis setup, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2509.22944", "https://github.com/huawei-csl/SINQ",
            "Separated from calibration-free SINQ to prevent mixing A0 and A1 evidence.",
        ),
        entry(
            "srr", "SRR", "Structured Residual Reconstruction PTQ", "A", "A1",
            "weight-only PTQ preserving dominant activation-scaled singular directions with low-rank residual repair",
            "calibration statistics/closed-form decomposition without end-to-end training in the PTQ core",
            "quantized residual representation and low-rank repair factors",
            "preserve selected activation-sensitive singular directions and reconstruct remaining error",
            "base codec; both low-rank factors and dtype/scales; selected-rank metadata; padding/header/alignment",
            "yes_if_protocol_matched",
            "use PTQ core without QPEFT; match calibration, ranks, factor precision, actual bytes, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2602.02001", "https://github.com/cyoonjun/srr",
            "The optional QPEFT route would belong in lane C and is not conflated with this row.",
        ),
        entry(
            "rescomp", "ResComp", "residual compensation PTQ", "A", "A1",
            "weight-only PTQ with sequential output-aligned residual compensation",
            "calibration forward/statistical correction; no gradient claim is used for this registry row",
            "quantized weights plus any compensation state surviving export",
            "align each quantized step with the original full-precision output using compensation-aware error",
            "ordinary codec; every non-folded compensation tensor; order/scheme metadata; padding/header/alignment",
            "yes_if_protocol_matched",
            "verify whether compensation is truly folded and measure exported bytes; match calibration, codec, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2604.07955", "https://github.com/list0830/ResComp",
            "No zero-overhead claim is assumed: exported artifacts decide whether compensation payload survives.",
        ),
        entry(
            "foem", "FOEM", "first-order error-aware compensation", "A", "A1",
            "weight-only PTQ with first-order-aware error compensation",
            "no explicit model backpropagation; uses latent/full-weight difference and Hessian-like calibration surrogate",
            "progressively compensated latent weights hardened into the final quantized weights; the paper algorithm adds no separate deployed FOEM state",
            "approximate loss with both first-order and second-order perturbation terms",
            "base codes/scales/zero-points; grouping/order metadata; padding/header/alignment; charge implementation-specific compensation only if an exporter actually serializes non-folded state",
            "yes_if_protocol_matched",
            "match calibration, codec, serialized compensation, actual bytes, and evaluation; report first-order diagnostics separately",
            "official_code_available; third_party_integration_available; external_reproduction_pending",
            "https://arxiv.org/abs/2507.11017", "https://github.com/Xingyu-Zheng/FOEM",
            "The paper-designated repository is Xingyu-Zheng/FOEM; ModelCloud/GPTQModel is only a secondary integration. FOEM is a relevant control because a Hessian-only proxy can miss the linear term.",
        ),
        entry(
            "yaqa", "YAQA", "full-model KL adaptive rounding", "B", "B",
            "weight PTQ with full-model KL Kronecker curvature sketches and fixed-point adaptive rounding",
            "backward/HVP-like Hessian-sketch collection may use calibration KL information, but the quantized representation is produced by fixed-point/LDL-style rounding without gradient learning; dense base stays frozen",
            "hardened quantized codes selected inside the underlying quantizer; no learnable YAQA rounding state survives export",
            "minimize full-model output KL under a structured curvature approximation",
            "only the underlying quantizer payload: codes/scales/zero-points, its required transform/codebook/decoder metadata, and padding/header/alignment; YAQA Hessian sketches, curvature factors and rounding search state are encoder-side and absent at inference",
            "yes_if_protocol_matched",
            "match calibration corpus/tokens, Hessian sketch type/rank and HVP cost, fixed-point/LDL rounding protocol, underlying quantizer, actual exported bytes, compute, and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2505.22988", "https://github.com/Cornell-RelaxML/yaqa-quantization",
            "Placed in B because collecting full-model KL curvature can require backward/HVP work; this is gradient-assisted statistics, not gradient learning or fine-tuning.",
        ),
        entry(
            "interplay_sq", "Effective Interplay S+Q", "max-scaled quantization plus magnitude sparsity", "A", "A0",
            "post-training sparsity/quantization interaction theory with LLM and vision experiments",
            "the core order/error analysis uses deterministic magnitude sparsity and max-scaled numerical encoding without calibration gradients",
            "a sparse quantized tensor; no separate repair state is introduced by the core analysis",
            "decompose the compounded dot-product error and characterize when sparsity and quantization are non-orthogonal and order dependent",
            "quantized values/scales; sparse values and mask/index or N:M metadata; ordering/format descriptors; padding/header/alignment",
            "mechanism_only",
            "use as the closest interaction-theory prior; a numerical comparison must match sparsity structure, quantizer, order, actual serialized bytes, model/tensors and evaluation",
            "primary_sources_verified; official_code_available; external_reproduction_pending",
            "https://openreview.net/forum?id=wJv4AIt4sK", "https://github.com/parsa-epfl/quantization-sparsity-interplay",
            "ICLR 2025 Spotlight already proves and measures Q/S non-orthogonality and order effects, and the author project page links the official implementation. The current repository may claim a different signed PSD-cross-term and physical-byte audit, not first interaction or orthogonality analysis.",
        ),
        entry(
            "choi_ecsq", "Hessian ECSQ", "Hessian-weighted quantization plus Huffman/ECSQ", "B", "B",
            "network weight quantization under a compression-ratio or entropy constraint",
            "diagonal task-Hessian weighting requires second-order/backward information; the paper also studies post-quantization fine-tuning, which must be separated from the core codec comparison",
            "cluster/code indices, reproduction values/codebook and a variable-length binary code",
            "minimize Hessian-weighted distortion subject to rate through entropy-constrained scalar quantization",
            "variable-length codewords or Huffman tree/code lengths; codebook/LUT; tensor shapes; scales if used; all headers, alignment and padding",
            "mechanism_only",
            "historical non-LLM scope; match Hessian computation, codebook, entropy coder, complete file bytes and exclude fine-tuning before using it as a frozen-compression comparator",
            "primary_sources_verified; no_official_repo_found; literature_only; external_reproduction_pending",
            "https://arxiv.org/abs/1612.01543", "not_found_in_primary_sources",
            "ICLR 2017 already combines Hessian-weighted distortion, a compression constraint and Huffman/ECSQ. Therefore Hessian-plus-rate or Hessian-plus-entropy is not a base novelty claim.",
        ),
        entry(
            "optimal_formats_entropy", "Optimal Formats", "variable-length weight formats", "A", "A0",
            "weight-format design and lossless coding after scalar quantization",
            "distributional format analysis and direct-cast quantization do not require model training",
            "quantized values encoded by a variable-length code and the decoder format state",
            "minimize squared error or KL-linked distortion at an expected code length",
            "actual code stream; symbol table/codebook or canonical code lengths; block/outlier descriptors; tensor metadata; headers/alignment/padding",
            "yes_if_protocol_matched",
            "match source weights, format family, entropy coder, decode state, complete serialized bytes, tensor coverage and evaluation",
            "primary_sources_verified; no_official_repo_found; external_reproduction_pending",
            "https://arxiv.org/abs/2505.12988", "not_found_in_primary_sources",
            "The paper shows that uniform quantization followed by variable-length lossless coding is optimal in its stated model. It reports average/sample code length rather than this repository's self-describing artifact bytes.",
        ),
        entry(
            "optimal_formats_fisher", "Optimal Formats", "Fisher layer-wise bit allocation", "B", "B",
            "mixed-rate allocation across model parameter tensors",
            "Fisher information links output KL to weighted parameter distortion and can require loss-gradient statistics",
            "per-tensor quantized streams and the selected bit-width/format allocation",
            "minimize Fisher-weighted distortion under an aggregate expected-rate constraint",
            "every per-tensor stream; bit-width/format identifiers; scales/codebooks; Fisher statistics only if required at decode; headers/alignment/padding",
            "mechanism_only",
            "match the Fisher corpus/objective and backward cost, candidate formats, aggregate physical bytes, tensor scope and evaluation",
            "primary_sources_verified; no_official_repo_found; external_reproduction_pending",
            "https://arxiv.org/abs/2505.12988", "not_found_in_primary_sources",
            "Separated from the paper's data-free format analysis because Fisher allocation is a gradient/statistics-assisted rate allocator.",
        ),
        entry(
            "projq", "ProjQ", "orthogonal projection for quantization and low-rank adapters", "C", "C",
            "adapter-aware quantization whose error is shaped toward a low-rank subspace before downstream adaptation",
            "the main evaluation includes low-rank adaptation/fine-tuning; the alternating projection mechanism must be isolated before any frozen-PTQ comparison",
            "quantized base plus low-rank adapter/correction factors and any projection state that is not folded",
            "push quantization noise into the adapter-correctable subspace while reducing the orthogonal residual",
            "base quantizer payload; both low-rank factors; projection/basis state unless regenerated or folded; adapter metadata; headers/alignment/padding",
            "mechanism_only",
            "separate frozen compensation from downstream LoRA, match rank/training budget, projection state, actual artifact bytes, model/tensors and evaluation",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2606.00494", "https://github.com/yy9301/ProjQ",
            "ProjQ already uses orthogonal subspace projection for quantization plus low-rank adaptation, so orthogonality-based low-rank coordination is not a first claim.",
        ),
        entry(
            "paretoq", "ParetoQ", "low-bit QAT scaling-law study", "C", "C",
            "quantization-aware fine-tuning/training across low-bit regimes",
            "gradient-based QAT or fine-tuning",
            "model and quantizer parameters",
            "task/distillation loss under quantization simulation",
            "exported codes/scales/zero-points; retained quantizer/adaptation state; unquantized tensors; padding/header/alignment",
            "mechanism_only",
            "use as a trained upper-bound/frontier with training compute disclosed, not as a strict frozen-PTQ control",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2502.02631", "https://github.com/facebookresearch/ParetoQ",
            "Training regime, rather than deployment bit width, determines lane C.",
        ),
        entry(
            "q_vdit", "Q-VDiT", "video-DiT weight/activation quantization with temporal distillation", "D", "D",
            "video diffusion Transformer quantization and distillation",
            "calibration and temporal-maintenance distillation optimize a video-generation objective; this is not the current frozen decoder-only LLM weight endpoint",
            "quantized video-DiT weights/activations plus estimator or distillation-trained state retained by export",
            "preserve token/feature information and spatiotemporal consistency under video-DiT quantization",
            "weight codes/scales; activation quantizer state; token-aware estimator state if retained; unquantized exceptions; runtime kernels/workspace; headers/padding/alignment",
            "no_scope_mismatch",
            "compare only in a video-DiT W/A deployment track with matched model, prompts, frames, denoising protocol, hardware, exported bytes, peak memory, latency, and video quality",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2505.22167", "https://github.com/wlfeng0509/Q-VDiT",
            "Primary sources describe a video-generation PTQ/distillation system. It informs the multimodal extension but is not a direct weight-only language-model PTQ baseline.",
        ),
        entry(
            "s2q_vdit", "S2Q-VDiT", "salient calibration and sparse-token distillation for W/A quantization", "D", "D",
            "video diffusion Transformer weight/activation quantization",
            "Hessian-aware calibration selection and attention-guided sparse-token distillation are encoder/training procedures",
            "quantized video-DiT weights and activation quantizer state; sparse-token selection is not asserted as runtime sparse attention",
            "improve W/A PTQ calibration and distillation for long video token sequences",
            "weight codes/scales; activation quantizer state; retained calibration/distillation state if any; unquantized exceptions; runtime kernels/workspace; headers/padding/alignment",
            "no_scope_mismatch",
            "keep training-time sparse-token distillation separate from runtime sparse attention; match video model, W/A scope, calibration, prompts, frames, steps, bytes, memory, latency, and quality",
            "official_repository_placeholder; code_not_released_as_of_2026-07-16; external_reproduction_pending",
            "https://arxiv.org/abs/2508.04016", "https://github.com/wlfeng0509/s2q-vdit",
            "The paper reports W4A6 video-DiT PTQ. Sparse tokens guide distillation during calibration and must not be counted as deployed sparse-attention acceleration.",
        ),
        entry(
            "quantsparse", "QuantSparse", "joint model quantization and attention sparsification", "D", "D",
            "HunyuanVideo-class video diffusion Transformer W/A quantization plus runtime sparse attention",
            "multi-scale salient attention distillation and second-order sparse-attention reparameterization jointly correct coupled quantization/sparsity error",
            "quantized model state, sparse-attention policy/reparameterization state, and any retained distillation-derived parameters",
            "jointly preserve global attention structure and local salient information under quantization and attention sparsity",
            "weight/activation quantizer payload; sparse support/index or routing state; reparameterization state; dense fallbacks; kernels/workspace; headers/padding/alignment",
            "no_scope_mismatch",
            "use a matched video-generation system protocol and report storage, runtime state, peak memory, attention-only and end-to-end latency separately; never multiply standalone speedups",
            "official_repository_placeholder; code_not_released_as_of_2026-07-16; external_reproduction_pending",
            "https://arxiv.org/abs/2509.23681", "https://github.com/wlfeng0509/QuantSparse",
            "This is the closest joint video quantization-plus-sparse-attention precedent. The official repository contains paper assets and says code will be released soon, so paper numbers remain literature-reported.",
        ),
        entry(
            "teacache", "TeaCache", "timestep-embedding-aware feature caching", "D", "D",
            "training-free temporal cache for video diffusion inference",
            "no model training in the core method; timestep-modulated input differences and rescaling decide cache reuse",
            "cached intermediate outputs and cache-control thresholds/statistics at runtime",
            "skip redundant denoising computation while bounding timestep-dependent output change",
            "cache buffers; cache keys/timestep state; thresholds/rescaling parameters; fallback compute; kernels/workspace; peak runtime memory",
            "no_scope_mismatch",
            "cache latency and memory must be measured under the same video model, prompts, frames, denoising steps, hardware, warmup, and quality protocol",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2411.19108", "https://github.com/ali-vilab/TeaCache",
            "TeaCache is a cache-only video diffusion system reference, not a stored-weight compression endpoint.",
        ),
        entry(
            "sparse_videogen", "Sparse VideoGen", "dynamic spatial-temporal sparse attention", "D", "D",
            "training-free runtime sparse attention for video diffusion Transformers",
            "online profiling classifies heads and predicts dynamic spatial or temporal sparse patterns; no dense-weight PTQ is implied",
            "head classifiers/profiling state, sparse layouts, and custom sparse-attention kernels",
            "preserve attention output while exploiting dynamic spatial-versus-temporal head structure",
            "sparse layouts/indices; profiling and classifier state; dense fallback; layout transforms; kernels/workspace; peak runtime memory",
            "no_scope_mismatch",
            "report attention-only and end-to-end latency separately with matched sequence length, hardware, kernel version, prompts, frames, steps, and quality",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2502.01776", "https://github.com/svg-project/Sparse-VideoGen",
            "Dynamic head type and online profiling make this a runtime attention-system comparison rather than a fixed weight-codec row.",
        ),
        entry(
            "sparse_vdit", "Sparse-vDiT", "offline searched diagonal, multi-diagonal, and vertical-stripe attention", "D", "D",
            "structured sparse attention and hardware-aware search for video diffusion Transformers",
            "offline search selects per-head/layer sparse strategies; the method changes runtime attention rather than only stored weights",
            "per-head pattern assignments, fused sparse layouts, search-selected schedule, and custom kernels",
            "minimize hardware cost while preserving video attention under recurring structured sparsity",
            "pattern descriptors; sparse layouts/indices; skipped-head metadata; dense fallback; kernels/workspace; peak runtime memory",
            "no_scope_mismatch",
            "match model, head/layer schedule, sequence geometry, hardware cost model, kernels, prompts, frames, steps, and video quality; distinguish theoretical FLOPs from measured latency",
            "primary_sources_verified; no_official_repo_found; external_reproduction_pending",
            "https://arxiv.org/abs/2506.03065", "not_found_in_primary_sources",
            "The primary paper reports model-specific structured patterns and measured video inference speedups, but no official repository was confirmed in the audited primary sources.",
        ),
        entry(
            "cachequant", "CacheQuant", "joint diffusion caching and quantization", "D", "D",
            "training-free image diffusion cache plus model quantization",
            "dynamic programming chooses a cache schedule and decoupled error correction addresses coupled and accumulated cache/quantization error",
            "quantized model state, cache schedule/controller state, correction state, and runtime cache buffers",
            "jointly optimize temporal reuse and structural quantization rather than independently stacking them",
            "quantized codes/scales; cache buffers and schedule; correction/controller state; unquantized exceptions; kernels/workspace; headers/padding/alignment",
            "no_scope_mismatch",
            "use as interaction-aware stacking precedent; image-diffusion results require new video-model validation with matched storage, cache memory, latency, and quality accounting",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2503.01323", "https://github.com/BienLuky/CacheQuant",
            "CacheQuant directly shows that independently optimized cache and quantization errors are not orthogonal. Its measured scope is image diffusion, not HunyuanVideo.",
        ),
        entry(
            "vmonarch", "VMonarch", "structured Monarch attention for video diffusion", "D", "D",
            "structured attention replacement for video diffusion Transformers",
            "alternating minimization, recomputation, and minimal tuning fit structured Monarch attention; this changes the model/runtime attention operator",
            "Monarch factors, update/recomputation state, and fused attention kernels",
            "capture intra-frame and inter-frame dynamic sparsity with a sub-quadratic structured matrix",
            "all Monarch factors; update state retained at inference; position/layout metadata; dense fallbacks; kernels/workspace; peak runtime memory",
            "no_scope_mismatch",
            "separate attention FLOPs and attention-kernel speed from end-to-end generation speed; match tuning budget, model, sequence geometry, hardware, and quality",
            "primary_sources_verified; no_official_repo_found; external_reproduction_pending",
            "https://arxiv.org/abs/2601.22275", "not_found_in_primary_sources",
            "VMonarch is evidence for structured local/global attention, but its reported attention-compute gains are not an end-to-end compression result.",
        ),
        entry(
            "monarchrt", "MonarchRT", "periodic structured, dynamic sparse, and dense-mixing attention", "D", "D",
            "finetuned structured attention for real-time autoregressive video diffusion",
            "finetuning and custom kernels fit a tiled Monarch parameterization to periodic position structure, dynamic sparse semantics, and dense mixing",
            "trained Monarch factors, tiled layout metadata, and custom Triton kernels",
            "retain expressive mixed attention structure while reducing real-time video attention cost",
            "trained factors; tiling/position metadata; dense-mixing state; kernels/workspace; peak runtime memory; model checkpoint delta",
            "no_scope_mismatch",
            "report kernel and full-pipeline speed separately; match finetuning data/steps, autoregressive regime, model, hardware, frames, and quality",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2602.12271", "https://github.com/Infini-AI-Lab/MonarchRT",
            "The paper argues that sparse semantics alone are insufficient because periodic structure and dense mixing remain. Kernel speedups must not be presented as end-to-end multipliers.",
        ),
        entry(
            "ropeslr", "RoPeSLR", "3D-RoPE sparse semantic spikes plus low-rank background", "D", "D",
            "sparse-low-rank attention replacement for long-sequence video diffusion Transformers",
            "head-wise low-rank parameterization and learned 3D absolute positional injection preserve RoPE-aware structure alongside sparse semantic spikes",
            "sparse routes/support, low-rank attention factors, positional parameters, and runtime kernels",
            "decompose attention into high-frequency sparse semantics and an extreme low-rank background continuum",
            "sparse support/index or routing state; low-rank factors; positional parameters; dense fallbacks; kernels/workspace; peak runtime memory",
            "no_scope_mismatch",
            "match sequence length, sparsity/rank, model, training or calibration budget, hardware, kernels, prompts, frames, steps, and quality; verify held-out transfer of the decomposition",
            "primary_sources_verified; no_official_repo_found; external_reproduction_pending",
            "https://arxiv.org/abs/2605.20659", "not_found_in_primary_sources",
            "RoPeSLR is the closest external sparse-plus-low-rank attention analogue to the proposed spectral gate, but it is not a weight Q/S/L artifact comparison.",
        ),
        entry(
            "bitnet_b158", "BitNet b1.58", "native ternary model training", "D", "D",
            "model architecture/training designed natively for ternary weights",
            "full pretraining or training from scratch in the low-bit architecture",
            "entire native low-bit model",
            "language-model training objective with ternary-weight constraints",
            "ternary weight encoding; scales; non-ternary tensors; architecture/runtime metadata; padding/header/alignment",
            "no_scope_mismatch",
            "compare only in a separately trained-model track; it is not post-hoc compression of the same pretrained checkpoint",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2402.17764", "https://github.com/microsoft/BitNet",
            "Native low-bit training cannot establish orthogonality or repair gains for fixed-checkpoint PTQ.",
        ),
        entry(
            "quest", "QuEST", "native/QAT low-bit weights and activations", "D", "D",
            "training-aware low-bit weights-and-activations system",
            "gradient-based native or quantization-aware training",
            "trained low-bit model and activation quantizers",
            "training loss under low-bit weight/activation simulation",
            "weight codes/scales; activation quantizer state; non-quantized tensors; runtime/kernel requirements; padding/header/alignment",
            "no_scope_mismatch",
            "use a separately trained W/A track; do not treat as frozen weight-only PTQ at equal nominal bits",
            "official_code_available; external_reproduction_pending",
            "https://arxiv.org/abs/2502.05003", "https://github.com/IST-DASLab/QuEST",
            "Lane D captures both training-regime and compressed-object mismatch with the current weight-only PTQ probe.",
        ),
    ]
    validate_rows(rows)
    return rows


def _is_source(value: str) -> bool:
    return value.startswith("https://") or value in URL_SENTINELS


def validate_rows(rows: Iterable[dict[str, str]]) -> None:
    """Fail fast if the registry violates its comparison contract."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("method matrix must not be empty")
    ids: set[str] = set()
    for index, row in enumerate(materialized, start=1):
        missing = [column for column in METHOD_COLUMNS if not str(row.get(column, "")).strip()]
        if missing:
            raise ValueError(f"row {index} has empty required fields: {missing}")
        if set(row) != set(METHOD_COLUMNS):
            raise ValueError(f"row {index} has unexpected schema: {sorted(set(row) ^ set(METHOD_COLUMNS))}")
        if row["method_id"] in ids:
            raise ValueError(f"duplicate method_id: {row['method_id']}")
        ids.add(row["method_id"])
        if row["lane"] not in {"A", "B", "C", "D"}:
            raise ValueError(f"invalid lane for {row['method_id']}: {row['lane']}")
        expected_lane = row["sub_lane"][0]
        if row["sub_lane"] not in LANE_DEFINITIONS or row["lane"] != expected_lane:
            raise ValueError(f"incompatible lane/sub_lane for {row['method_id']}")
        if row["strict_ptq_direct_comparison"] not in ALLOWED_COMPARISON:
            raise ValueError(f"invalid comparison value for {row['method_id']}")
        if not _is_source(row["primary_paper"]) or not _is_source(row["official_repo"]):
            raise ValueError(f"invalid source field for {row['method_id']}")

    required = {
        "RTN", "GPTQ", "AWQ", "SparseGPT", "Wanda", "SpQR", "SqueezeLLM",
        "LQER", "QERA", "EoRA", "QuIP#", "QTIP", "AQLM", "OmniQuant",
        "SpinQuant", "SliderQuant", "LiftQuant", "D²Quant", "ADMM-Q", "HAS-VQ",
        "SEPTQ", "HESTIA", "AAAC", "DAQ",
        "Q-Palette", "SLiM", "OBR", "EfficientQAT", "LLM-QAT", "TurboQuant",
        "SharQ", "Joint structural pruning + MPQ", "Q-VDiT", "S2Q-VDiT",
        "QuantSparse", "TeaCache", "Sparse VideoGen", "Sparse-vDiT", "CacheQuant",
        "VMonarch", "MonarchRT", "RoPeSLR",
    }
    present = {row["method"] for row in materialized}
    if missing_methods := required - present:
        raise ValueError(f"required methods missing: {sorted(missing_methods)}")

    sliders = [row for row in materialized if row["method"] == "SliderQuant"]
    expected_slider_variants = {
        "default; fused channel scales and rank-4 LoRA",
        "SliderQuant+ with runtime rotations",
    }
    if len(sliders) != 2 or {row["variant"] for row in sliders} != expected_slider_variants:
        raise ValueError("SliderQuant must have separate default and SliderQuant+ rows")
    if {row["sub_lane"] for row in sliders} != {"B"}:
        raise ValueError("both SliderQuant calibration variants must remain in lane B")

    palettes = [row for row in materialized if row["method"] == "Q-Palette"]
    if len(palettes) != 2 or {row["sub_lane"] for row in palettes} != {"A0", "A1"}:
        raise ValueError("Q-Palette must have separate data-free A0 and data-aware A1 rows")

    foem = next(row for row in materialized if row["method_id"] == "foem")
    if foem["official_repo"] != "https://github.com/Xingyu-Zheng/FOEM":
        raise ValueError("FOEM official_repo must point to the paper-designated repository")

    yaqa = next(row for row in materialized if row["method_id"] == "yaqa")
    if "absent at inference" not in yaqa["payload_must_count"]:
        raise ValueError("YAQA must distinguish encoder-side curvature from deployment payload")

    lift = [row for row in materialized if row["method"] == "LiftQuant"]
    if len(lift) != 2 or {row["lane"] for row in lift} != {"B", "C"}:
        raise ValueError("LiftQuant must have exactly one lane-B row and one lane-C row")
    block = next(row for row in lift if row["lane"] == "B")
    e2e = next(row for row in lift if row["lane"] == "C")
    for fragment in ("4096", "2048", "2 epochs", "STE", "T*=M T^-1"):
        if fragment not in " ".join(block.values()):
            raise ValueError(f"LiftQuant block row is missing protocol fragment: {fragment}")
    for fragment in ("4096", "1 epoch", "cross-entropy"):
        if fragment not in " ".join(e2e.values()):
            raise ValueError(f"LiftQuant E2E row is missing protocol fragment: {fragment}")
    if "never double-count" not in block["payload_must_count"]:
        raise ValueError("LiftQuant payload must make fused and unfused transform representations exclusive")


def render_csv(rows: list[dict[str, str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=METHOD_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_summary(rows: list[dict[str, str]]) -> str:
    lane_counts = Counter(row["lane"] for row in rows)
    sub_lane_counts = Counter(row["sub_lane"] for row in rows)
    no_repo = [row["method"] for row in rows if row["official_repo"] == "not_found_in_primary_sources"]
    lines = [
        "# 外部压缩方法分层对比矩阵",
        "",
        f"> 生成日期与来源核验截止：{VERIFIED_AS_OF}。本页由 `scripts/build_compression_method_matrix.py` 生成。",
        "",
        "## 这张表是什么、又不是什么",
        "",
        "这是一个**文献协议矩阵（literature matrix）**，用于在实验前固定方法范围、优化信号、可变状态、真实载荷和可比边界；它**不是本仓库已经测得的结果表**。`reproduction_status` 明确区分“有官方代码”和“已在本仓库复现”，当前行不得据此推断精度或速度。",
        "",
        "矩阵不填未经一手来源核验的精度、PPL、吞吐、GPU 小时或优胜结论。LiftQuant 的样本长度、epoch 与变换开销来自论文/README 协议；固定 commit 的 README flags 与两阶段 parser 并不一一对应，不能写成已验证的 executable protocol。仓库另有一层 compatibility-patched smoke，但没有 PPL/任务精度，不能进入方法排名。",
        "",
        "## 分层",
        "",
        f"共 {len(rows)} 行：A={lane_counts['A']}、B={lane_counts['B']}、C={lane_counts['C']}、D={lane_counts['D']}；细分 A0={sub_lane_counts['A0']}、A1={sub_lane_counts['A1']}、B={sub_lane_counts['B']}、C={sub_lane_counts['C']}、D={sub_lane_counts['D']}。",
        "",
        "- **A0**：冻结稠密模型，data-free，无校准样本/激活统计，无反向传播。",
        "- **A1**：冻结稠密模型，允许校准样本、激活统计或闭式二阶量，但无反向传播。",
        "- **B**：冻结 dense pretrained base；允许为局部 calibration reconstruction、曲率统计执行 STE/backward/HVP，或优化最终可融合的量化/辅助状态，但不做全局任务恢复。",
        "- **C**：使用全局 CE/KL 梯度优化、QAT、下游 PEFT/微调，或主结果流水线以任务损失恢复全模型行为。",
        "- **D**：压缩对象或训练范式不一致，如 KV-cache、仅激活、原生低比特训练和结构/NAS 搜索。",
        "",
        "## 严格直接比较规则",
        "",
        "`yes_if_protocol_matched` 不是自动可比；至少要匹配模型/检查点、压缩对象、校准数据与 token 数、评测集、实际部署载荷和 kernel 可执行性。B 类还必须记录 backward/HVP/STE 次数、是否实际学习参数、更新变量、GPU-hours 与峰值显存。C 类可以作为允许全局训练的前沿或机制参照，但不能用于证明冻结 PTQ 的严格优势。D 类单列，不把其名义压缩率混进权重 PTQ 横轴。",
        "",
        "同压缩率必须使用实际 artifact 字节，而不是名义 bit 或熵下界。统一计费形式为：",
        "",
        "`actual_payload_bytes = codes + scales/zero_points + transforms + codebooks + masks/indices + low_rank + biases + scheme_metadata + padding + headers/alignment + uncompressed_exceptions`",
        "",
        "只有在导出产物中真实折叠、且解码/执行不再需要独立参数时，才能把某个 correction/scale/transform 计为零额外字节。除载荷外，统一记录 calibration tokens、steps/epochs、是否 backprop、updated variables、GPU-hours、peak memory 与 kernel/backend。",
        "",
        "## LiftQuant 必须拆成两行",
        "",
        "- **Block correction（B）**：论文/README 写为 STE、4096 个 RedPajama 样本 × 2048 token、2 epochs；固定 commit parser 只有 `nsamples1/2` 与 `epochs1/2`，机械映射还会把每阶段 4096 减为 3968，语义尚未由可执行协议确认。更新 binary `Wq` 与 fused `T*=M T^-1`，目标为 block output MSE，不使用全局 CE。部署必须计 `1-bit lifted codes + scales + [fused T* OR (M + inverse-whitening factors)] + padding/alignment`；两种等价表示不能重复计费。",
        "- **Optional E2E（C）**：4096 个样本 × 4096 token，1 epoch CE，继续更新连续量化参数。论文主结果采用 E2E；仓库将其标为可选，并推荐部署优先使用 block correction。",
        "",
        "固定 commit `72b3875` 的未补丁 Qwen2.5 layer-0 smoke 在 Transformers 4.57 上因 `Catcher.attention_type` 缺失失败；一行透明 metadata 兼容补丁后，8-window、1+1 epoch、单层 smoke 以 exit 0 写出 artifact。它只证明 bounded control flow，不含 PPL/任务精度、全模型压缩或部署 payload。完整证据见 `results/liftquant_official_integration_20260714/`。",
        "",
        "## 需要显式拆分的其他协议",
        "",
        "- **SqueezeLLM 主 Fisher 路径（B）**：官方 from-scratch quantization 先在独立框架中计算目标模型 loss gradient-square checkpoint，再把它用于 sensitivity-weighted k-means。它不训练 dense 参数，但确实使用 backward，不能放入 no-backward A1。",
        "- **SpinQuant optimized rotation（C）**：官方 `optimize_rotation.py` 冻结 dense base 后，在 WikiText2 上用 Hugging Face `Trainer.train()` 和全局 causal-LM loss 优化 R1/R2。按目标函数属于全局恢复 C，而不是局部 reconstruction B。",
        "- **SliderQuant（B）**：默认路径在 128×2048-token 校准集上以 AdamW 优化 channel scales 与所有线性层的 rank-4 LoRA；默认 20 epochs，W2A16 为 60 epochs。两者在导出时吸收到权重。**SliderQuant+（B）** 另含不可吸收的 Hadamard 运行时变换，必须单列 transform、kernel、workspace 和 latency。",
        "- **Q-Palette A0/A1**：data-free codec/search 属 A0；使用 QTIP proxy Hessian 和 validation perplexity loss 做 mixed-scheme selection 的 data-aware 路径属 A1。二者都不 retrain，但校准预算和数据泄漏边界不同。",
        "- **YAQA（B）**：B 来自全模型 KL Hessian sketch 可能需要 backward/HVP；fixed-point/LDL rounding 本身不做梯度学习，Hessian 与 search state 不进入部署 payload。",
        "- **ADMM-Q / HAS-VQ / SEPTQ（A1）**：ADMM 闭式交替、k-means 和逐列二阶补偿都是冻结模型上的 encoder-side 局部迭代，不等于 backward/STE 微调。最终只免计确实不随 artifact 部署的 Hessian、分解和搜索状态；HAS-VQ 的 codebook+sparse residual 与 SEPTQ 的保留值+support 必须计费。",
        "- **HESTIA（C）**：离线 Hutch++ 只负责温度调度；随后在 10B token 上以 AdamW 优化全模型 causal-LM loss，明确属于 QAT，不得放进 frozen PTQ 排名。",
        "- **MXFP 协议参考**：arXiv:2601.09555 是 benchmark/format study 而非本矩阵新增实测方法。MXFP 比较必须计 block value stream、共享 E8M0 scale 和 padding，并单列其 3/4 pre-scale；不得把 MXFP4 的 nominal 4 bit 与 INT4 artifact 直接等同。",
        "",
        "## 方法索引",
        "",
        "| 方法 | 变体 | lane | 严格 PTQ 直接比较 | 复现状态 |",
        "|---|---|:---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {variant} | {sub_lane} | {direct} | {status} |".format(
                method=_markdown_cell(row["method"]),
                variant=_markdown_cell(row["variant"]),
                sub_lane=row["sub_lane"],
                direct=row["strict_ptq_direct_comparison"],
                status=_markdown_cell(row["reproduction_status"]),
            )
        )
    lines.extend(
        [
            "",
            "## Multimodal and video-system supplement",
            "",
            "The D-lane supplement records Q-VDiT, S2Q-VDiT, QuantSparse, TeaCache, Sparse VideoGen, Sparse-vDiT, CacheQuant, VMonarch, MonarchRT, and RoPeSLR. These rows cover video/image diffusion W/A quantization, cache state, runtime sparse attention, structured attention, or sparse-low-rank attention, so they are mechanism and stacking references rather than direct frozen LLM weight-PTQ controls.",
            "",
            "Reported speed, storage, and quality numbers remain in the dedicated multimodal strategy document instead of this registry. They use different models, prompts, frame counts, denoising steps, hardware, kernels, sparsity definitions, and training budgets. Standalone speedups must never be multiplied; a joint stack requires one measured end-to-end run plus separate storage, runtime-state, peak-memory, attention-only, and quality accounting.",
            "",
            "QuantSparse and S2Q-VDiT have official repository placeholders but no released implementation as of the verification date. Q-VDiT, TeaCache, Sparse VideoGen, CacheQuant, and MonarchRT have inspectable official code. Sparse-vDiT, VMonarch, and RoPeSLR are marked without a confirmed official repository in the audited primary sources.",
        ]
    )
    lines.extend(
        [
            "",
            "## 来源与复现边界",
            "",
            "每一行都给出 `primary_paper` 与 `official_repo` 字段；`not_found_in_primary_sources` 表示截至核验日未在一手来源中确认官方仓库，不等于证明仓库永远不存在。RTN 是定义性 baseline，因此使用 `not_applicable_baseline`。",
            "",
            "截至核验日未确认官方仓库的方法：" + "、".join(dict.fromkeys(no_repo)) + "。",
            "",
            "除本仓库已实现并用于既有 exploratory runs 的 RTN baseline 外，矩阵只把“官方代码可获得”记为可复现入口；外部方法在本仓库实际运行前不能标记为 measured/reproduced。完整逐行字段与来源见 `method_matrix.csv`。",
            "",
            "## 重新生成与防漂移",
            "",
            "```bash",
            "python scripts/build_compression_method_matrix.py",
            "python scripts/build_compression_method_matrix.py --check",
            "pytest -q tests/test_compression_method_matrix.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def expected_outputs() -> dict[str, str]:
    rows = build_rows()
    return {
        "method_matrix.csv": render_csv(rows),
        "summary.md": render_summary(rows),
    }


def _read_exact(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_exact(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def write_outputs(output_dir: Path, check: bool = False) -> list[Path]:
    outputs = expected_outputs()
    paths: list[Path] = []
    drift: list[str] = []
    for name, content in outputs.items():
        path = output_dir / name
        paths.append(path)
        if check:
            if not path.exists():
                drift.append(f"missing: {path}")
            elif _read_exact(path) != content:
                drift.append(f"out of date: {path}")
        else:
            _write_exact(path, content)
    if drift:
        raise SystemExit("generated method matrix drift detected:\n" + "\n".join(drift))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed outputs exactly match the generator without rewriting them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_outputs(args.output_dir, check=args.check)
    verb = "verified" if args.check else "wrote"
    for path in paths:
        print(f"{verb}: {path}")


if __name__ == "__main__":
    main()
