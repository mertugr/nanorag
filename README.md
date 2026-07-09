# nanorag

Local **RAG** orchestrator over owned libraries:

| Library | Role |
|---------|------|
| [tinyann](third_party/tinyann) | Vector search (HNSW / exact / …) |
| [nanollm](third_party/nanollm) | LLM generate / chat |
| **nanorag** | Chunk store, in-house embedder, retrieve → prompt → generate |

No Hugging Face or third-party ML runtimes on the default path.

## Status

- [x] CMake project linking tinyann + nanollm as libraries  
- [x] **Word2VecEmbedder** (`word2vec-v1`) — skip-gram + neg sampling, train-on-ingest, pure C++  
- [x] `HashingEmbedder` (`hashing-v1`) — still available via `--embedder hashing`  
- [x] Chunk store (TSV) + HNSW index save/load + `embeddings.nw2v`  
- [x] CLI: `smoke` · `ingest` · `ask` · `Retriever::open`  
- [x] Retrieval/grounding tests (cats / tinyann / water)  
- [x] Optional generation when `--model` / `--tokenizer` are set  
- [ ] TF–IDF hybrid / nanoembed encoder — later  

## Build

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Quick start

```bash
# Unit smoke (no index files)
./build/nanorag smoke

# Build an index from the demo corpus
./build/nanorag ingest --chunks data/demo/chunks.tsv --out index/demo --dim 64 --epochs 50

# Retrieve + print RAG prompt (no LLM)
./build/nanorag ask --index index/demo --query "What is tinyann?" --k 3
./build/nanorag ask --index index/demo --query "which animal meows" --k 2

# Retrieve + generate (needs a .nanollm checkpoint + tokenizer)
# ./build/nanollm is not built by default; build nanollm CLI separately or point at yours:
#   cmake -S third_party/nanollm -B build-nanollm -DNANOLLM_BUILD_CLI=ON
#   cmake --build build-nanollm --target nanollm
#   ./build-nanollm/nanollm write-synthetic --model-out models/tiny.nanollm --tokenizer-out models/tiny.nllmtok --context 2048
./build/nanorag ask --index index/demo --query "What is tinyann?" \
  --model models/tiny.nanollm --tokenizer models/tiny.nllmtok --max-tokens 64
```

## Index layout

```
index/demo/
  chunks.tsv           # id \t source \t text
  vectors.hnsw.tann    # tinyann HNSW binary
  embeddings.nw2v      # word2vec-v1 weights (when using default embedder)
  meta.txt             # embedder_id, dim, metric, …
```

Query-time embedder **must** match `embedder_id` / `dim` in `meta.txt`.

## Layout

```
include/nanorag/   # embedder, chunk_store, prompt, pipeline
src/main.cpp       # CLI
tests/             # CTest
third_party/       # git submodules: tinyann, nanollm
data/demo/         # sample chunks
```

## Roadmap (short)

1. **Phase 0** — scaffold (done)  
2. **Word2Vec embeddings** — train-on-ingest, tested (done)  
3. **Phase 1** — install/export polish for all libs  
4. **Phase 2** — hybrid keyword + fuller CLI  
5. **Phase 3** — nanoembed encoder tower  
6. **Phase 4** — production hardening + tagged release  

## Submodule notes

`tinyann` exposes `TINYANN_BUILD_CLI` / `TINYANN_BUILD_TESTS` (nanorag forces them OFF when nested).

## License

TBD (align with tinyann / nanollm before public release).
