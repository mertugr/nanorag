# nanorag

Local **C++17 RAG** orchestrator over owned libraries:

| Library | Role |
|---------|------|
| [tinyann](https://github.com/mertugr/tinyann) | In-memory ANN (HNSW, …) — git submodule |
| [nanollm](https://github.com/mertugr/nanollm) | From-scratch LLM inference — git submodule |
| **nanorag** | Chunk store, in-house embedders, retrieval, **grounded** answers |

No Hugging Face or third-party ML runtimes on the default path.

**Status:** Phase 0 sealed · Phase 1 packaging · Phase 2 eval foundation · contrastive-v2 dense encoder. Honest hard retrieval (no train synonym leak) is still weak — not production dual-encoder quality.

---

## Architecture

```
chunks (+ train pairs)
    → embedder (contrastive-v2 default)
    → tinyann HNSW index  +  in-memory BM25 (sparse)
query → retrieve (hybrid dense-primary by default)
      → answerability gate
    → refuse: "I don't know"
    → else: extractive evidence + [#id] citations
optional nanollm generate (validated; falls back if ungrounded)
```

**Retrieval modes** (`--retrieve dense|sparse|hybrid`, default **hybrid**): dense ANN, BM25, or dense-primary fusion with adaptive BM25 boost. See [docs/HYBRID_RETRIEVAL.md](docs/HYBRID_RETRIEVAL.md) for before/after eval and where hybrid helps vs not.

### Embedders

| ID | Learning | Use |
|----|----------|-----|
| `contrastive-v2` | InfoNCE dual encoder: word+char-ngram+bigram pool → MLP → L2 | **Default** |
| `contrastive-v1` | Legacy mean-pool BOW (load only) | Old indexes |
| `word2vec-v1` | Skip-gram on chunk text | Ablation |
| `hashing-v1` | Feature hashing | Ablation / tests |

Contrastive ingest injects a **`NO_EVIDENCE` sentinel** (`id=-1`) plus near-miss OOD queries so the model can rank “no answer” for out-of-scope questions.

### Grounding (shared `default_grounding_config()`)

1. **Score floor** (`min_score`)
2. **Answerability** — keep a passage if:
   - enough **query content tokens** appear in the passage, **or**
   - score ≥ `min_score_without_query_support` (paraphrase path)
3. **NO_EVIDENCE** top with no answerable real hit → refuse
4. Empty / blank query → refuse
5. Extractive answer quotes evidence and cites `[#id]` only
6. `validate_grounding` rejects illegal cites and unsupported content

Near-miss example (corpus has water, not alcohol):

```bash
./build/nanorag ask --index index/demo -q "What is the boiling point of alcohol?"
# → I don't know

./build/nanorag ask --index index/demo -q "At one atmosphere when does pure water become gas?"
# → Evidence from retrieved passages: … H2O … [#6]
```

Extractive mode is **evidence quoting**, not abstractive QA. Free-form generation is optional and must pass the same validator.

---

## Build (submodules — default / development)

```bash
git clone --recurse-submodules https://github.com/mertugr/nanorag.git
cd nanorag
# Submodules are public: github.com/mertugr/tinyann and github.com/mertugr/nanollm

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNANORAG_USE_SUBMODULES=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/nanorag smoke
```

### Install to a prefix (self-contained from submodules)

```bash
cmake --install build --prefix $HOME/nanorag-prefix
# bins:  $HOME/nanorag-prefix/bin/nanorag
# cmake: $HOME/nanorag-prefix/lib/cmake/{nanorag,tinyann,nanollm}
# headers under include/
```

### Optional: find_package path

Build and install **tinyann** / **nanollm** first (or use the prefix above), then:

```bash
cmake -S . -B build-pkg   -DNANORAG_USE_SUBMODULES=OFF   -DCMAKE_PREFIX_PATH=$HOME/nanorag-prefix
```

See [COMPATIBILITY.md](COMPATIBILITY.md) for version matrix and format versions.

---

## CLI

```bash
./build/nanorag smoke

./build/nanorag ingest \
  --chunks data/demo/chunks.tsv \
  --pairs  data/demo/train_pairs.tsv \
  --out index/demo \
  --embedder contrastive --dim 64 --epochs 200

./build/nanorag ask --index index/demo -q "…" [--min-score 0.25] [--retrieve hybrid]
# Generate mode is chat-aware (meta chat_family, stops, add_bos):
./build/nanorag ask --index index/demo -q "…" --mode generate \
  --model m.nanollm --tokenizer t.nllmtok --meta m.meta.txt \
  [--chat-family chatml|llama3|tinyllama|raw] [--allow-approx-chat] \
  [--system "…"] [--use-model-system] [--no-bos] [--temperature 0]
# Exact HF jinja: render with nanollm tools/hf_render_chat.py then --prompt-file
./build/nanorag eval-suite --data data/demo [--retrieve hybrid]
# Phase 2: R@k/MRR (dense/sparse/hybrid A/B) + refuse + grounding + ablations

./build/nanorag eval-paraphrase --index index/demo \
  --pairs data/demo/eval_paraphrase.tsv
# top hit among *real* docs only (skips NO_EVIDENCE)

./build/nanorag eval-grounding --index index/demo \
  --pairs data/demo/eval_paraphrase.tsv
# in-domain cites + near-miss refuse + blank + validator negatives
```

---

## Index layout

```
index/demo/
  chunks.tsv              # id \t source \t text  (may include id=-1 NO_EVIDENCE)
  vectors.hnsw.tann       # tinyann HNSW
  embeddings.nctr         # contrastive-v2 weights (or .nw2v for word2vec)
  meta.txt                # v2: embedder_id, dim, n_chunks, n_real_chunks, has_no_evidence
```

Query-time embedder **must** match `embedder_id` / `dim`.

---

## Data

```
data/demo/
  chunks.tsv              # neutral passages (not query-shaped answers)
  train_pairs.tsv         # TRAIN ONLY paraphrase_query \t positive_id
  eval/
    retrieval.tsv         # easy held-out (may share keywords) — smoke
    retrieval_hard.tsv    # hard: zero content-token overlap with gold
    refuse.tsv            # must answer exactly "I don't know"
    README.md             # format + disjointness rule
  eval_paraphrase.tsv     # deprecated thin pointer → eval/
  eval_hard_ood.tsv       # deprecated thin pointer → eval/
```

---


## Evaluation (Phase 2 foundation)

Labeled sets live under `data/demo/eval/` and **must not overlap** train queries
(`data/demo/train_pairs.tsv`). The harness enforces disjointness.

| Metric | Set | Meaning |
|--------|-----|---------|
| R@k / MRR **easy** | `eval/retrieval.tsv` | May share gold keywords — **smoke only** |
| R@k / MRR **hard** | `eval/retrieval_hard.tsv` | Zero gold keywords **and** no train synonym/near-dup leakage |
| Refuse pass rate | `eval/refuse.tsv` | Exact `I don't know`, empty context |
| Grounding full pass | easy retrieval labels | Answer + valid cites + gold cited |
| Ablations | easy + hard | hashing / word2vec / contrastive |

> Easy R@1≈1 is **not** dual-encoder quality. **Hard** forbids gold keyword overlap **and**
> train near-dups / same-id synonym tokens / morph n-gram leakage — that is the honest score.

```bash
# Full suite (also runs via ctest: nanorag_eval_harness)
./build/nanorag eval-suite --data data/demo
ctest --test-dir build -R eval_harness --output-on-failure
```

See `data/demo/eval/README.md`.

## Tests

| Target | Coverage |
|--------|----------|
| `nanorag_tests` | Hashing, store, contrastive paraphrase recall, ablations |
| `nanorag_grounding_tests` | Citations, alcohol near-miss refuse, blank query, OOD, validator |
| `nanorag_eval_harness` | Disjoint train/eval, R@k/MRR, refuse, grounding, ablations, hard cases |

---

## Known limits (intentional Phase 0 boundary)

- Bag-of-words mean embeddings — not a large dual encoder
- Quality needs good train pairs and near-miss OOD pairs
- **Host-endian** binary formats (`*.tann`, `.nctr`, `.nw2v`) — same machine endianness required (no endian marker)
- Single-threaded; no server yet
- Honest hard retrieval is still weak (see eval hard R@k) — not production dual-encoder quality

---

## Roadmap

1. **Phase 0** — scaffold + contrastive + grounding (**sealed**)
2. **Phase 1** — hybrid submodules + install/export, CI, VERSION, LICENSE (**done** / Milestone 0 hygiene)
3. **Phase 2** — eval foundation: labeled sets, R@k/MRR, refuse, grounding, ablations
4. **Phase 2+** — hybrid dense+BM25 retrieval (**this**); stronger embedders next
5. **Later** — generate/chat parity with nanollm, incremental index, HTTP serve
## License

MIT for **nanorag** — see [LICENSE](LICENSE).  
Submodules (also MIT, public):

- [tinyann](https://github.com/mertugr/tinyann) — MIT
- [nanollm](https://github.com/mertugr/nanollm) — MIT
