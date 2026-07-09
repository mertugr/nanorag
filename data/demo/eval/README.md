# Demo evaluation sets (Phase 2 foundation)

| File | Format | Purpose |
|------|--------|---------|
| `../train_pairs.tsv` | `query \\t gold_id` | Contrastive **training only** |
| `retrieval.tsv` | `query \\t gold_id` | Held-out retrieval (recall@k, MRR) |
| `refuse.tsv` | `query` | Must refuse with `I don't know` |

## Disjointness rule

After normalize (lowercase, collapse whitespace), **no query string** in
`retrieval.tsv` or `refuse.tsv` may appear in `train_pairs.tsv`.

The eval harness and unit tests enforce this and fail if violated.

## Metrics

- **Retrieval:** recall@k, MRR (real docs only; skip `NO_EVIDENCE` id=-1)
- **Refuse:** fraction of refuse-set queries with exact IDK + empty used context
- **Grounding:** for retrieval queries that answer, citation legality + gold id cited + validator ok
- **Ablations:** same retrieval metrics for hashing / word2vec / contrastive
