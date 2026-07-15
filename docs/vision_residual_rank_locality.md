# Vision Residual Streams, Effective Rank, and Attention Locality

Updated: 2026-07-16

## Research question

The useful connection is not simply "local attention has higher rank." A more precise question is:

> How does the spatial connectivity graph of attention control the rate and scale at which patch-token diversity is contracted, and how much genuinely new subspace is injected by attention and MLP residual updates?

For patch features `X_l in R^(N x d)`, a pre-norm Transformer block can be viewed schematically as:

```text
Delta_attn_l = Attn(LN(X_l))
Y_l          = X_l + Delta_attn_l
Delta_mlp_l  = MLP(LN(Y_l))
X_(l+1)      = Y_l + Delta_mlp_l
```

Softmax attention mixes tokens through a row-stochastic operator. Repeated dense mixing tends to amplify a shared/DC token mode and suppress token differences. Residual identity paths, LayerNorm, value projections, MLP updates, masks, and changing attention graphs all alter that simplified dynamics.

## What recent primary work establishes

1. [Anti-Oversmoothing in Deep Vision Transformers via the Fourier Domain Analysis](https://arxiv.org/abs/2203.05962) models self-attention as a low-pass operation and connects excessive depth to loss of high-frequency patch information.
2. [ContraNorm](https://arxiv.org/abs/2303.06562) treats oversmoothing as dimensional collapse and uses entropy effective rank to show that ViT token representations can lose dimensional diversity with depth.
3. [On the Role of Attention Masks and LayerNorm in Transformers](https://arxiv.org/abs/2405.18781) proves that pure masked attention can still collapse, while sparse or local masks can slow the collapse rate; it also shows LayerNorm makes the equilibrium picture richer than an unconditional rank-one claim.
4. [Mind the Gap](https://arxiv.org/abs/2410.07799) identifies a softmax-attention singular-value gap associated with rank collapse in width as context length grows.
5. [ResiDual Transformer Alignment with Spectral Decomposition](https://arxiv.org/abs/2411.00246), published in TMLR, reports intrinsically low-dimensional visual attention-head residuals whose principal components specialize by task or attribute.
6. [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) identifies high-norm artifact tokens used for internal/global computation. This means top spectral modes may reflect storage tokens rather than spatial semantics.
7. [Frequency-Aware Token Reduction](https://arxiv.org/abs/2511.21477), a NeurIPS 2025 paper, explicitly preserves high-frequency tokens to mitigate rank collapse and oversmoothing during token reduction.
8. [Locality-Attending Vision Transformer](https://arxiv.org/abs/2603.04892), an ICLR 2026 paper, adds a learned Gaussian spatial bias so patches favor neighbors while retaining global information, improving dense prediction in its experiments.
9. [Dynamics of the Transformer Residual Stream](https://arxiv.org/abs/2605.14258) is a 2026 LLM preprint, not a vision result, but its Jacobian-spectrum and network-topology framing suggests a useful cross-modal extension: couple representation spectra to the topology of the learned mixing graph.

## The locality/rank mechanism

### Global attention

Dense global attention creates a connected mixing graph in one layer. If its dominant shared mode is separated by a large spectral gap, non-shared token modes contract quickly. This can reduce token effective rank and spatial high-frequency energy even while channel covariance across many images remains high.

### Fixed local windows

Local masks break the graph into neighborhoods. They slow *global* homogenization because information cannot mix across distant patches in one step. This does not mean local collapse disappears: tokens can become similar inside each window while different windows remain distinct. A global effective-rank number can therefore look healthy while within-window diversity has already collapsed.

### Shifted windows and sparse global links

Window shifts, dilation, register/CLS tokens, or sparse global edges connect previously separate components. Across depth they increase receptive field and alter the attention graph's spectral gap. The expected behavior is two-stage mixing: fast smoothing inside a neighborhood, followed by slower cross-neighborhood smoothing.

### Residual paths

`X_(l+1) = X_l + Delta_l` can keep `X_(l+1)` high-rank even when `Delta_l` is low-rank. Consequently, residual-stream effective rank alone can hide attention collapse. The important measurement is the rank and orientation of the update relative to the existing residual subspace.

## Measurements needed for a valid vision experiment

Do not pool all patches and images into one covariance and call the result "the effective rank." Record four distinct objects:

| Object | Matrix | Question |
|---|---|---|
| Per-image token representation | centered `X_l` with shape `N x d` | Are patches becoming linearly redundant within one image? |
| Dataset activation covariance | samples/patches by channel | How many channel directions vary over the dataset? |
| Attention operator | `A_l` with shape `N x N` per head | How concentrated is token mixing and what is its spectral gap? |
| Residual update | `Delta_attn_l`, `Delta_mlp_l` | How much new subspace does each branch inject? |

For each layer/head, collect:

- global token effective rank and stable/participation rank;
- within-window effective rank and between-window centroid rank;
- attention spectral entropy, effective rank, `sigma_2 / sigma_1`, and leading singular-value gap;
- expected spatial attention distance and mass inside radii 1, 2, and 4 patches;
- 2D DCT or graph-Laplacian high-frequency energy of patch features;
- principal-angle/CCA overlap between `X_l` and each residual update;
- update novelty after projecting `Delta_l` outside the top residual principal subspace;
- high-norm token count, identity, and spectral leverage, with CLS/register tokens reported separately;
- task metrics for classification and dense prediction, because global invariance and local detail prefer different spectra.

## Minimal matched experiment

Use the same image subset, preprocessing, layer fractions, and estimator for four model types:

| Model family | Connectivity role |
|---|---|
| DeiT/ViT | Dense global-attention baseline |
| Swin | Local windows plus shifted cross-window mixing |
| DINOv2 with/without registers | Tests whether high-norm storage tokens dominate top modes |
| LocAtViT | Learned soft locality rather than a hard window mask |

Recommended first pass:

- 256 natural images plus matched patch-shuffled and phase-scrambled controls;
- early, middle, and late layers;
- `resid_pre`, per-head `attn_out`, merged `attn_out`, `mlp_out`, and `resid_post`;
- pretrained and random-init variants where architecture permits;
- bootstrap confidence intervals over images, not over pooled patches;
- shared-axis depth curves and paired pretrained/random deltas.

## Falsifiable hypotheses

1. **Local masks slow global rank collapse.** Swin should retain higher global token effective rank than dense ViT at matched depth, but may show lower within-window rank before shifted-window mixing.
2. **Residual rank masks update collapse.** `resid_post` rank can remain high while `attn_out` rank and update novelty fall sharply with depth.
3. **Connectivity predicts contraction.** Larger attention-graph spectral gaps should correlate with faster loss of non-DC/high-frequency feature energy.
4. **Registers isolate global modes.** Register-equipped models should move high spectral leverage away from spatial patches and produce smoother patch spectra; this must be checked rather than assumed universal.
5. **Dense tasks need local spectral diversity.** Segmentation quality should correlate more strongly with within-neighborhood rank and high-frequency retention than classification accuracy does.
6. **Head specialization is low-dimensional but useful.** Low head-residual rank is not automatically collapse; if principal components are stable and task-selective, low rank can represent specialization rather than redundancy.

## Decision criterion

The vision extension is worthwhile if connectivity-aware metrics explain behavior that pooled covariance effective rank cannot. The strongest result would be a matched relationship of the form:

```text
attention topology -> operator spectral gap -> local/global rank trajectory
                   -> residual-update novelty -> dense/classification performance
```

A simple correlation between effective rank and accuracy is insufficient because rank may reflect useful specialization, register artifacts, or unstructured noise.
