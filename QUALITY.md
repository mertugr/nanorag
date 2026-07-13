# Quality report — nanorag

**Date:** 2026-07-13  
**Version:** 0.1.0 (`main`)  
**Stack:** tinyann ≥ 0.1.0 · nanollm ≥ 0.4.0 · see [COMPATIBILITY.md](COMPATIBILITY.md)

This document is the honest quality bar for retrieval, refuse, and grounding.
It is **not** a dual-encoder leaderboard claim.

---

## How we measure quality

| Split | File | What it measures |
|-------|------|------------------|
| **Easy** | `data/demo/eval/retrieval.tsv` | Held-out paraphrases; may share gold keywords — **smoke**, not hard semantic quality |
| **Hard** | `data/demo/eval/retrieval_hard.tsv` | **Zero content-token overlap with gold**; train/easy/refuse disjoint; synonym-leak + morph n-gram integrity — **honest** semantic retrieval |
| **Refuse** | `data/demo/eval/refuse.tsv` | OOD / near-miss → exact “I don't know”, empty `used` |
| **Grounding** | easy labels | Extractive answerability + citation validator |

Integrity is enforced in `include/nanorag/eval.hpp` (`assert_zero_keyword_overlap`, `assert_hard_no_synonym_leak`, train/eval disjointness).  
**No train/hard leak** is allowed when reporting hard metrics.

Default eval recipe: `./build/nanorag eval-suite --data data/demo`  
(contrastive train: dim=128, epochs=420, seed=7, hard_neg_k=5, hard_neg_loss_weight=0.15 — see `src/main.cpp` `cmd_eval_suite`).

---

## Current quality state (`main`, 2026-07-13)

### Baseline (`eval-suite --no-ablations`, hybrid primary)

| Metric | Value | Notes |
|--------|-------|--------|
| Easy R@1 / R@3 / R@5 / MRR | **1.00** / 1.00 / 1.00 / 1.00 | Smoke ceiling on tiny set (n=18) |
| Hard R@1 / R@3 / R@5 / MRR | **0.222** / 0.389 / 0.611 / 0.356 | Honest bar (n=18) → **4/18** at R@1 |
| Hard dense | same as hybrid | BM25 weak on zero-overlap hard |
| Hard sparse R@1 | ~0.056 | Expected ~0 by construction |
| Easy sparse R@1 | ~0.944 | Lexical safety net |
| Refuse pass rate | **1.00** | n=16 |
| Grounding full pass | **1.00** | On easy labels |
| Integrity | **PASS** | `disjoint_ok`, `hard_zero_keyword_ok` |

**Gates:** all quality gates **PASS**.

### What this means in product terms

| Claim | Supported? |
|-------|------------|
| Local extractive RAG on a small demo corpus works | **Yes** |
| Near-miss / OOD refuse is reliable on the demo refuse set | **Yes** |
| Hybrid helps keyword-aligned easy retrieval | **Yes** (vs pure dense hashing ablations historically) |
| Strong zero-overlap paraphrase / metaphor retrieval | **No** — hard R@1 ≈ 0.17–0.28 |
| Production dual-encoder quality | **No** |

Easy R@1 = 1 is necessary for smoke tests; it is **not** evidence of hard semantic retrieval.

---

## Hard-climb experiment (integrity-safe hyperparam + train expansion)

**Artifact:** [results/hard_climb_results.xlsx](results/hard_climb_results.xlsx) (176 completed configs)

Attempted goal: hard R@1 ≥ **0.60** without integrity leaks.

| Result | Value |
|--------|-------|
| Completed runs | **176** |
| Best hard R@1 | **0.2778** (5/18) |
| Mean hard R@1 | **~0.189** |
| Hard R@1 histogram | 2/18: 24 · 3/18: 80 · 4/18: 48 · **5/18: 24** |
| Target 0.60 | **Not reached** |
| Easy / refuse on best | **1.0 / 1.0** |

**Best config (by R@1 then MRR):**

```
name:   d192_e480_s3_nw0.75_hn0.12_k5
dim:    192
epochs: 480
seed:   3
ngram_w: 0.75
hard_neg_w: 0.12
hard_neg_k: 5
n_train: 146 (integrity-filtered expansion)
hard R@1 / R@5 / MRR: 0.278 / 0.667 / 0.393
```

### Conclusion from the climb

Under the **current hard set** and a **bag-of-tokens + char n-gram + bigram contrastive** embedder, hard R@1 is **stuck near 0.28**. More seed×weight grids will not reach 0.6.

The hard set forbids:

- content-token overlap with gold  
- same-id train distinctive-token leak of hard-query words  
- high morph n-gram coverage vs gold  

So queries like “purring sofa buddy” cannot share tokens with the Felis catus passage **or** with cat train paraphrases that use those distinctive words. Geometry alone is too weak for BOW-class models.

---

## Default embedder (today)

| ID | Type | Notes |
|----|------|--------|
| **contrastive-v2** (default) | InfoNCE, token + char n-grams + bigrams, L2 | Train-time hard-neg mining optional |
| contrastive-v1 | Legacy | Load-only |
| word2vec-v1 / hashing-v1 | Ablations | Not production quality drivers |

This is an **owned, dependency-free** path — not a sentence-transformer dual encoder.

---

## Next step: better text model for embeddings

To move hard R@1 substantially (toward 0.5–0.6+ on the **same** integrity hard set), change the **embedding model class**, not only hyperparameters:

1. **Dual-encoder / encoder-only transformer** (small, CPU-friendly) trained or distilled on paraphrase pairs  
   - Keep train/hard integrity gates unchanged  
   - Export to a nanorag-native weight format (or ONNX/GGUF later if needed)  
2. **Or** accept hard R@1 ~0.2–0.3 as the honesty bound for BOW and ship extractive RAG on lexical + modest semantic signal  

Recommended product path:

- **Ship now:** extractive + hybrid + refuse (current main)  
- **Invest next:** small dual-encoder embedder with the same eval harness and **no leak** policy  

---

## How to re-run quality

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNANORAG_USE_SUBMODULES=ON
cmake --build build -j
./build/nanorag eval-suite --data data/demo --no-ablations
# Full ablations:
./build/nanorag eval-suite --data data/demo
```

Hard-climb spreadsheet (historical): `results/hard_climb_results.xlsx`.

---

## Related docs

- [COMPATIBILITY.md](COMPATIBILITY.md) — versions and formats  
- [docs/HYBRID_RETRIEVAL.md](docs/HYBRID_RETRIEVAL.md)  
- [docs/EMBEDDER_HARDNEG.md](docs/EMBEDDER_HARDNEG.md)  
