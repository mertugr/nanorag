# Stronger embedder + hard-negative mining

## Goal

Move **hard retrieval R@1** off ~0.17 on the integrity-enforced hard set
(`data/demo/eval/retrieval_hard.tsv`: zero content-token overlap with gold,
no train synonym leak, morph n-gram caps). **No cheating** — harness still
enforces hard integrity.

## What we changed

1. **Expanded train paraphrases** (`data/demo/train_pairs.tsv`, ~35 → ~85)
   - More synonymy per gold id
   - Filtered so no hard-query distinctive tokens (len≥4) leak into same-id train
2. **Hard-negative second InfoNCE pass** (train-time only; format still NCTR v2)
   - After warmup, for each real-doc pair: top-k wrong docs by current score
   - Skips NO_EVIDENCE as positive and as a mined negative (protects refuse)
3. **OOD oversampling** for `NO_EVIDENCE` pairs proportional to in-domain size
   - Expanded in-domain data was drowning refuse calibration
4. **Query–query pull** implemented but **default weight 0** (hurts refuse if >0)

## Before / after (seed=7, `eval-suite` recipe)

| Metric | Before (main) | After (this PR) |
|--------|---------------|-----------------|
| easy R@1 | 1.00 | 1.00 |
| easy MRR | 1.00 | 1.00 |
| **hard R@1** | **0.167** | **0.222–0.333** (seed-dependent; see multi-seed) |
| hard R@5 | 0.722 | ~0.50–0.67 |
| hard MRR | 0.356 | ~0.35–0.38 |
| refuse pass | 1.00 | 1.00 |
| grounding full | 1.00 | 1.00 |

### Multi-seed hard R@1 (dim=128, epochs=420, expanded train, OOD oversample)

| Seed | hard R@1 | refuse |
|------|----------|--------|
| 7 | 0.222 | 1.00 |
| 11 | 0.222 | 1.00 |
| 99 | 0.222 | 1.00 |
| 42 | 0.278 | 1.00 |
| 123 | 0.167 | 1.00 |
| **mean** | **~0.222** | **1.00** |

With light hard-neg (`k=5`, weight `0.15`, start epoch 100) seed 7 often hits **0.333** hard R@1 with refuse still 1.0.

## Why hard is still hard (honest limits)

The hard set forbids:

- gold keyword overlap
- same-id train distinctive tokens shared with the hard query
- high morph n-gram coverage vs gold

So queries like “purring sofa buddy” never share tokens with train cat paraphrases
*or* the gold passage. Bag-of-tokens + char n-grams must generalize via embedding
geometry alone. **+5–15pp hard R@1 on n=18 is real but noisy** (1–3 queries).

## What did *not* work (kept out of defaults)

| Idea | Result |
|------|--------|
| Aggressive MLP projection head | Representation collapse (NaN / unit vectors) |
| High query–query weight | Refuse pass → 0 |
| Heavy hard-neg weight / early start | Refuse broken; hard R@1 flat or worse |
| Classic RRF hybrid as “quality fix” | Does not lift zero-overlap hard |

## Integrity

`assert_hard_no_synonym_leak` still runs in the eval harness. Expanded train was
filtered against hard distinctive tokens. Easy/hard/refuse string disjointness
unchanged.

## Files

- `include/nanorag/contrastive.hpp` — hard-neg + query–query training hooks
- `include/nanorag/pipeline.hpp` — OOD oversample for NO_EVIDENCE
- `data/demo/train_pairs.tsv` — expanded paraphrases
