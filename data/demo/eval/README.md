# Demo evaluation sets (Phase 2 foundation)

| File | Format | Purpose |
|------|--------|---------|
| `../train_pairs.tsv` | `query \t gold_id` | Contrastive **training only** |
| `retrieval.tsv` | `query \t gold_id` | **Easy** held-out retrieval (may share gold keywords) |
| `retrieval_hard.tsv` | `query \t gold_id` | **Hard** held-out: **zero content-token overlap** with gold |
| `refuse.tsv` | `query` | Must refuse with `I don't know` |

## Why two retrieval splits?

Easy paraphrases often keep distinctive gold words (`H2O`, `feathers`, `hierarchical`…).
Bag-of-words / hashing embedders then look strong (**R@1≈1**) without true semantic match.
That number is **not** a dual-encoder quality measure.

`retrieval_hard.tsv` forbids any shared **content token** (same stop list + `simple_tokenize`
as answerability) between query and gold passage. Report **hard** R@k / MRR as the honest
retrieval score. **contrastive-v2** (projection MLP + char n-grams + bigrams + synonymy
train pairs) targets this split; hashing/word2vec remain weak ablations.

## Disjointness rules

After normalize (lowercase, collapse whitespace):

1. No query in any eval file may appear in `train_pairs.tsv`
2. Hard queries must not appear in the easy set either
3. Hard queries must have `keyword_overlap(query, gold_text) == {}` (enforced by harness)

## Metrics

- **retrieval_easy:** R@k, MRR — smoke / lexical path only
- **retrieval_hard:** R@k, MRR — honest semantic retrieval
- **Refuse:** exact IDK + empty used context
- **Grounding:** easy labels — citation legality + gold cited + validator
- **Ablations:** hashing / word2vec / contrastive on **both** easy and hard
