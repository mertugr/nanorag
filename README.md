# nanorag

Local **RAG** orchestrator over owned libraries only:

| Library | Role |
|---------|------|
| [tinyann](third_party/tinyann) | Vector search (HNSW / exact / …) |
| [nanollm](third_party/nanollm) | LLM generate / chat |
| **nanorag** | Chunk store, **in-house embedders**, retrieve → prompt → generate |

No Hugging Face or third-party ML runtimes on the default path.

## Honest status on retrieval

**Unsupervised Word2Vec / hashing are not “real semantic retrieval.”**  
They mostly reward token overlap. Early demos that only queried with shared keywords (and even stuffed answer-shaped lines into the corpus) were invalid as paraphrase tests.

What works for paraphrase retrieval in this repo today:

| Embedder | How it learns | Paraphrase? |
|----------|----------------|-------------|
| `contrastive-v1` (**default**) | Supervised InfoNCE on `(query → doc_id)` pairs you provide | **Yes** (tested) |
| `word2vec-v1` | Skip-gram on chunk text only | Weak / ablation |
| `hashing-v1` | Feature hashing | Keyword only |

**You must supply train pairs** for contrastive ingest. Documents stay neutral facts; questions live in pair files—not inside the chunks.


## Grounding & citations

Answers are **gated** before any generation:

1. Retrieve candidates (tinyann)
2. Per-passage **answerability** gate:
   - keep if embedding score is high **and** query content tokens overlap the passage, **or**
   - score alone is very high (trained paraphrase path: `min_score_without_query_support`)
   - `NO_EVIDENCE` sentinel / near-miss training → refuse
3. If nothing answerable → exact `I don't know` (no LLM call)
4. Else **extractive** answer from kept passages only, each cited as `[#id]`
5. Optional LLM: accepted only if `validate_grounding` passes; else extractive fallback

Near-miss examples that must refuse (corpus has water, not alcohol):

```bash
./build/nanorag ask --index index/demo \
  -q "What is the boiling point of alcohol?" --min-score 0.20
# → I don't know   (must NOT return cats / planets / water)

./build/nanorag ask --index index/demo \
  -q "At one atmosphere when does pure water become gas?" --min-score 0.20
# → H2O … 373.15 K [#6]

./build/nanorag eval-grounding --index index/demo \
  --pairs data/demo/eval_paraphrase.tsv --min-score 0.20
# in-domain cites · alcohol/ethanol/iron/… refuse · validator negatives
```

`nanorag_grounding_tests` includes the alcohol near-miss case (not only absurd OOD).

## Build

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Quick start

```bash
./build/nanorag smoke

# Train contrastive embedder on pairs, index with tinyann HNSW
./build/nanorag ingest \
  --chunks data/demo/chunks.tsv \
  --pairs  data/demo/train_pairs.tsv \
  --out index/demo \
  --embedder contrastive --dim 64 --epochs 200

# Held-out paraphrases: enforces low token Jaccard(query, gold) then recall@1
./build/nanorag eval-paraphrase \
  --index index/demo \
  --pairs data/demo/eval_paraphrase.tsv \
  --max-jaccard 0.30

./build/nanorag ask --index index/demo \
  -q "Which feline housemate purrs when comfortable?" --k 3
```

## Data layout

```
data/demo/
  chunks.tsv           # neutral passages (id, source, text) — not query-shaped
  train_pairs.tsv      # paraphrase_query \t positive_id  (contrastive train)
  eval_paraphrase.tsv  # held-out paraphrases for eval-paraphrase

index/demo/
  chunks.tsv
  vectors.hnsw.tann
  embeddings.nctr      # contrastive-v1 weights
  meta.txt
```

## Tests (what “works” means)

`ctest` checks:

1. **Paraphrase gate** — eval queries have low token Jaccard vs gold text  
2. **Contrastive recall@1 = 6/6** on held-out paraphrases  
3. **Grounding** — RAG prompt contains the retrieved passage text  
4. **Ablation** — word2vec and hashing get **&lt; 6/6** on the same hard paraphrase set  

CLI `eval-paraphrase` fails if any query/gold pair exceeds `--max-jaccard` (default 0.25–0.30).

## Roadmap

1. Phase 0 scaffold — done  
2. Contrastive paraphrase embeddings + honest eval — done  
3. Phase 1 — install/export polish for tinyann / nanollm / nanorag  
4. Stronger encoders (deeper tower / nanoembed) still pure C++  
5. Production hardening  

## License

TBD (align with tinyann / nanollm before public release).
