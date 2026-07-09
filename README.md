# nanorag

Local **RAG** orchestrator over owned libraries:

| Library | Role |
|---------|------|
| [tinyann](third_party/tinyann) | Vector search (HNSW / exact / …) |
| [nanollm](third_party/nanollm) | LLM generate / chat |
| **nanorag** | Chunk store, in-house embedder, retrieve → prompt → generate |

No Hugging Face or third-party ML runtimes on the default path.

## Phase 0 status

- [x] CMake project linking tinyann + nanollm as libraries  
- [x] `HashingEmbedder` (pure C++, `hashing-v1`)  
- [x] Chunk store (TSV) + HNSW index save/load  
- [x] CLI: `smoke` · `ingest` · `ask`  
- [x] Optional generation when `--model` / `--tokenizer` are set  
- [ ] Better embedders (TF–IDF / Word2Vec / nanoembed) — later phases  

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
./build/nanorag ingest --chunks data/demo/chunks.tsv --out index/demo --dim 512

# Retrieve + print RAG prompt (no LLM)
./build/nanorag ask --index index/demo --query "What is tinyann?" --k 3

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

1. **Phase 0** — this scaffold (done)  
2. **Phase 1** — install/export polish for all libs  
3. **Phase 2** — fuller CLI, docs, hybrid keyword  
4. **Phase 3** — self-trained embedders (Word2Vec → nanoembed)  
5. **Phase 4** — production hardening + tagged release  

## Submodule notes

`third_party/tinyann` includes a small CMake patch (not yet on upstream `main` until you push it):

- `TINYANN_BUILD_CLI` / `TINYANN_BUILD_TESTS` (default ON standalone; nanorag sets them OFF)

Push that change to the tinyann repo, then update the submodule pointer.

## License

TBD (align with tinyann / nanollm before public release).
