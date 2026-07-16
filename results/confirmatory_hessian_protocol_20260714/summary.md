# Confirmatory Hessian-repair protocol (2026-07-14)

This is a preregistered data/split manifest, not a model result. The builder loads only a pinned local tokenizer and local WikiText cache; it has no download or text fallback path.

## Fixed design

- Model/tokenizer: `EleutherAI/pythia-70m` snapshot `a39f36b100fe8a5377810d56c3f4789b9c53ac42`.
- Seeds: `[17, 29, 43, 59, 71, 89, 101, 113]`.
- Calibration: `32 x 256` train tokens per seed; `256` pairwise source-row-disjoint windows in total.
- Validation: `32 x 256` fixed validation windows.
- Test: `64 x 256` fixed test windows.
- Epsilon grid (13 points): `0, 0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1`.
- Positive local linear+quadratic fit points: `0.03125, 0.0625, 0.09375, 0.125`.

## Leakage controls

- Exact content uses `Unicode NFKC, lowercase, then collapse all whitespace runs to one ASCII space` before SHA256 deduplication.
- Tokenization uses `native raw dataset text without normalization`; normalization never changes the model/PPL input.
- Deduplication priority is test, validation, then train, so duplicate calibration content cannot displace held-out evidence.
- Token 5-gram set-Jaccard rejects pairs at `>= 0.8`; `61776` final pairs were checked and `0` violations remain.
- Maximum retained-window Jaccard: `0.041322314`.
- Each native source row is consumed by at most one candidate window, including rejected candidates; seed allocations therefore cannot share source rows.

## Privacy/reproducibility boundary

`protocol.json` stores only native row IDs, SHA256 of normalized content, token lengths, and allocation ranges. It stores neither source text nor token IDs. Dataset fingerprints and the pinned tokenizer snapshot identify the required local inputs.
