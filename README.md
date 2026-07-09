# nanorag

Local **C++17 RAG** orchestrator over owned libraries:

| Library | Role |
|---------|------|
| [tinyann](https://github.com/mertugr/tinyann) | In-memory ANN (HNSW, …) — git submodule |
| [nanollm](https://github.com/mertugr/nanollm) | From-scratch LLM inference — git submodule |
| **nanorag** | Chunk store, in-house embedders, retrieval, **grounded** answers |

No Hugging Face or third-party ML runtimes on the default path.

**Status:** Phase 0 sealed. **Phase 1 packaging** — hybrid submodules (default) + install/export + CI. Not yet a production dual-encoder product.

---

## Architecture

```
chunks (+ train pairs)
    → embedder (contrastive-v1 default)
    → tinyann HNSW index
query → retrieve → answerability gate
    → refuse: "I don't know"
    → else: extractive evidence + [#id] citations
optional nanollm generate (validated; falls back if ungrounded)
```

### Embedders

| ID | Learning | Use |
|----|----------|-----|
| `contrastive-v1` | Supervised InfoNCE on `(query → doc_id)` pairs | **Default** |
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
# nanollm is private — auth required for that submodule

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

./build/nanorag ask --index index/demo -q "…" [--min-score 0.25]
./build/nanorag ask --index index/demo -q "…" --mode generate \
  --model m.nanollm --tokenizer t.nllmtok

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
  embeddings.nctr         # contrastive-v1 weights (or .nw2v for word2vec)
  meta.txt                # v2: embedder_id, dim, n_chunks, n_real_chunks, has_no_evidence
```

Query-time embedder **must** match `embedder_id` / `dim`.

---

## Data

```
data/demo/
  chunks.tsv              # neutral passages (not query-shaped answers)
  train_pairs.tsv         # paraphrase_query \t positive_id
  eval_paraphrase.tsv     # held-out paraphrases
  eval_hard_ood.tsv       # realistic near-miss / OOD questions
```

---

## Tests

| Target | Coverage |
|--------|----------|
| `nanorag_tests` | Hashing, store, contrastive paraphrase recall, ablations |
| `nanorag_grounding_tests` | Citations, alcohol near-miss refuse, blank query, OOD, validator |

---

## Known limits (intentional Phase 0 boundary)

- Bag-of-words mean embeddings — not a large dual encoder
- Quality needs good train pairs and near-miss OOD pairs
- Host-endian binary formats (same class as tinyann)
- Single-threaded; no server yet
- License TBD (align with tinyann / nanollm before public release)

---

## Roadmap

1. **Phase 0** — scaffold + contrastive + grounding (**sealed**)
2. **Phase 1** — hybrid submodules + install/export, CI, VERSION, LICENSE (**this**)
3. **Phase 2+** — stronger embedders, hybrid lexical+dense, production packaging

## License

MIT for **nanorag** sources — see [LICENSE](LICENSE).  
Submodules: **tinyann** (MIT); **nanollm** (see that repo; may still be TBD).
