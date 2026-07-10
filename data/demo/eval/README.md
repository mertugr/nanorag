# Demo evaluation sets (Phase 2 foundation)

| File | Format | Purpose |
|------|--------|---------|
| `../train_pairs.tsv` | `query \t gold_id` | Contrastive **training only** |
| `retrieval.tsv` | `query \t gold_id` | **Easy** held-out (may share gold keywords) — smoke |
| `retrieval_hard.tsv` | `query \t gold_id` | **Hard** held-out with full integrity (below) |
| `refuse.tsv` | `query` | Must refuse with `I don't know` |

## Why two retrieval splits?

Easy paraphrases often keep distinctive gold words. Bag-of-words then looks strong
without true semantic match. **Do not treat easy R@1 as dual-encoder quality.**

## Hard integrity (synonym-leak resistant)

`retrieval_hard.tsv` must pass **all** of the following (enforced by harness):

1. **Zero content-token overlap** with gold passage (`simple_tokenize` + stop list)
2. **String disjoint** from train + easy + refuse (normalized)
3. **Train near-dup ban**: max sequence similarity to any train query **< 0.50**
4. **Train token Jaccard ban**: content-token Jaccard to any train query **< 0.25**
5. **Same-id train token ban**: no distinctive token (len≥4) shared with any train
   query labeled to the same gold id (blocks synonym memorization)
6. **Morph n-gram ban**: fraction of query content char-ngrams (3/4) found in gold
   **≤ 0.15** (blocks feline↔felis style leakage for n-gram models)

If a reported hard R@1 jump required near-duplicate train paraphrases of the hard
set, that gain is **invalid**. Fix the data, not the model score.

## Metrics

- **retrieval_easy:** R@k, MRR — smoke / lexical path only
- **retrieval_hard:** R@k, MRR — honest semantic retrieval under integrity rules
- **Refuse / grounding / ablations:** as before
