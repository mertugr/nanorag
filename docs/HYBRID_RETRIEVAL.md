# Hybrid retrieval (dense + BM25)

## What it is

Default query path is **hybrid** (`RetrieveMode::Hybrid`):

1. **Dense** — contrastive (or other) embedder + tinyann HNSW cosine
2. **Sparse** — in-memory Okapi BM25 over the same `ChunkStore` (no extra index files)
3. **Fusion** — dense-primary: re-rank the dense candidate pool with BM25 only when
   there is a *strong* lexical hit; optionally inject sparse-only docs with high BM25

CLI: `--retrieve dense|sparse|hybrid` (default `hybrid`) on `ask`, `eval-*`, `eval-suite`.

## Design constraints (learned on demo eval)

| Rule | Why |
|------|-----|
| Dense-primary fusion (not classic RRF by default) | RRF with weak BM25 **hurt** hard R@1 and OOD refuse |
| Adaptive: fuse only if `max BM25 ≥ strong_sparse_inject` (~2.5) | Weak partial matches (e.g. token `one`) must not re-rank |
| Never BM25-index `NO_EVIDENCE` (id=-1) | Lexical ranking must not promote the refuse sentinel |
| Report **dense cosine** as hit `score` after hybrid rank | Inflated hybrid scores broke `min_score_without_query_support` and caused false answers on OOD |
| Strong inject threshold | Sparse-only inject only when BM25 is confident |

## Before / after (demo `data/demo`, seed=7, contrastive-v2)

Measured with `./build/nanorag eval-suite --data data/demo`.

### Before (dense-only primary, pre-hybrid)

| Split | R@1 | R@3 | R@5 | MRR |
|-------|-----|-----|-----|-----|
| easy | 1.00 | 1.00 | 1.00 | 1.00 |
| hard | 0.167 | 0.50 | 0.722 | 0.356 |
| refuse pass | 1.00 | | | |
| grounding full | 1.00 | | | |

### After (hybrid primary + A/B in report)

| Mode | easy R@1 | easy MRR | hard R@1 | hard R@5 | hard MRR | refuse |
|------|----------|----------|----------|----------|----------|--------|
| **hybrid (primary)** | **1.00** | **1.00** | **0.167** | **0.722** | **0.356** | **1.00** |
| dense | 1.00 | 1.00 | 0.167 | 0.722 | 0.356 | 1.00 |
| sparse (BM25) | 0.944 | 0.958 | 0.056 | 0.222 | 0.116 | — |

Ablation of interest:

| Embedder + mode | easy R@1 | hard R@1 |
|-----------------|----------|----------|
| hashing + dense | 0.889 | 0.056 |
| **hashing + hybrid** | **0.944** | 0.056 |
| contrastive + dense/hybrid | 1.00 | 0.167 |

## Where hybrid **helps**

1. **Weak dense embedders + keyword queries**  
   Hashing easy R@1: **0.889 → 0.944** with hybrid. BM25 recovers keyword-aligned docs when bag-of-hashing fails.

2. **Real corpora with mixed lexical + semantic queries**  
   When the user reuses document terms (IDs, product names, error codes), strong BM25 can inject or promote the right passage even if the dual encoder ranks it mid-list.

3. **Easy / smoke sets with gold keyword overlap**  
   Sparse alone already gets easy R@1 ≈ 0.94; hybrid keeps dense’s perfect easy score while retaining a lexical safety net.

4. **Debugging & ablations**  
   Eval report always prints dense / sparse / hybrid side-by-side so regressions are visible.

## Where hybrid **does not help** (on this demo)

1. **Hard retrieval (`eval/retrieval_hard.tsv`)**  
   By construction: **zero content-token overlap** with gold (and synonym-leak bans).  
   BM25 on gold ≈ 0 → adaptive path falls back to dense → **hard metrics identical to dense** (R@1 0.167).  
   Hybrid is not a substitute for a stronger dual encoder.

2. **Pure paraphrase / synonym queries**  
   Same reason: no shared tokens → no BM25 signal.

3. **OOD refuse**  
   After score-report fix, refuse pass stays 1.0. Classic RRF or score-inflating fusion **hurt** refuse (false answerable passages). Do not reintroduce score inflation.

4. **Already-perfect dense easy**  
   Contrastive easy R@1 is already 1.0; hybrid cannot improve the smoke ceiling on this tiny set.

## Operational guidance

| Use | Flag |
|-----|------|
| Production default | `--retrieve hybrid` (default) |
| Ablate / debug encoder | `--retrieve dense` |
| Keyword-only baseline | `--retrieve sparse` |
| Eval A/B | `eval-suite` always reports dense + sparse + primary |

## Implementation map

| Piece | Location |
|-------|----------|
| BM25 + fusion | `include/nanorag/hybrid.hpp` |
| Retriever wiring | `include/nanorag/pipeline.hpp` (`retrieve`, `set_retrieve_mode`) |
| Eval A/B + gates | `include/nanorag/eval.hpp` |
| CLI | `src/main.cpp` (`--retrieve`) |

## Next quality steps (not this PR)

- Larger hard set + raise hard R@1 gates after encoder work
- Hard-negative mining / deeper dual encoder
- Optional cross-encoder re-rank on top-k
- Learned fusion weights on a held-out set
