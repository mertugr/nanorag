# Compatibility

**Library version:** `0.1.0` (see [VERSION](VERSION), `CMakeLists.txt` `project(... VERSION ...)`).

nanorag is the **product / orchestrator** over two owned libraries:

```
tinyann 0.1.x  ──┐
                 ├──►  nanorag 0.1.x
nanollm 0.4.x  ──┘
```

## Version matrix

| Component | Version | How nanorag consumes it |
|-----------|---------|-------------------------|
| **nanorag** | **0.1.0** | this repo |
| **tinyann** | **≥ 0.1.0** | submodule `third_party/tinyann` **or** `find_package(tinyann 0.1)` |
| **nanollm** | **≥ 0.4.0** | submodule `third_party/nanollm` **or** `find_package(nanollm 0.4)` |

CMake defaults: `-DNANORAG_USE_SUBMODULES=ON` (development).  
Package path: `-DNANORAG_USE_SUBMODULES=OFF -DCMAKE_PREFIX_PATH=...`.

| Stack release | tinyann | nanollm | Notes |
|---------------|---------|---------|--------|
| nanorag **0.1.x** | ≥ 0.1.0 | ≥ 0.4.0 | Phase 0 seal + packaging + hybrid + grounding fixes |

Always prefer **git submodule SHAs** checked into this repo for bit-identical builds; package versions are minimums for `find_package`.

## Formats (nanorag-owned)

| Artifact | Version field | Notes |
|----------|---------------|--------|
| `meta.txt` | `nanorag_index_meta_version=2` | `n_real_chunks`, `has_no_evidence`; load validates kind/metric/sizes |
| `embeddings.nctr` | NCTR **v2** | contrastive-v2 (ngrams+bigrams; hard-neg train-time only); **v1 still loadable** |
| `embeddings.nw2v` | NW2V **v1** | word2vec-v1 |
| chunks | TSV | `id \t source \t text` (id `-1` reserved for `NO_EVIDENCE` sentinel) |

## Formats (from dependencies)

| Artifact | From | Notes |
|----------|------|--------|
| `vectors.hnsw.tann` | tinyann | HNSW graph; **host-endian**; cosine for nanorag indexes |
| `.nanollm` / `.nllmtok` | nanollm | Optional generate path; see nanollm COMPATIBILITY (format **v7** / tok **v4**) |

## Endianness

All multi-byte binary artifacts (`*.tann`, `.nctr`, `.nw2v`, `.nanollm`, `.nllmtok`) are **host-endian**.
Do not copy indexes/models between opposite-endian machines without conversion.
Same-endian cross-arch (e.g. arm64 LE ↔ x86_64 LE) is fine.

## Development (default)

```bash
git clone --recurse-submodules https://github.com/mertugr/nanorag.git
cd nanorag
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNANORAG_USE_SUBMODULES=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Installed packages

```bash
# Install each lib (example prefix)
cmake -S /path/to/tinyann -B build-tinyann -DTINYANN_BUILD_TESTS=OFF -DTINYANN_BUILD_CLI=OFF
cmake --build build-tinyann && cmake --install build-tinyann --prefix $PREFIX

cmake -S /path/to/nanollm -B build-nanollm -DNANOLLM_BUILD_TESTS=OFF -DNANOLLM_BUILD_CLI=OFF
cmake --build build-nanollm && cmake --install build-nanollm --prefix $PREFIX

cmake -S . -B build -DNANORAG_USE_SUBMODULES=OFF -DCMAKE_PREFIX_PATH=$PREFIX
cmake --build build && cmake --install build --prefix $PREFIX
```

Self-contained submodule install also re-exports tinyann + nanollm CMake packages under the same prefix.

## Related docs

- [tinyann COMPATIBILITY.md](https://github.com/mertugr/tinyann/blob/main/COMPATIBILITY.md)
- [nanollm COMPATIBILITY.md](https://github.com/mertugr/nanollm/blob/main/COMPATIBILITY.md)
- [docs/HYBRID_RETRIEVAL.md](docs/HYBRID_RETRIEVAL.md)
- [docs/EMBEDDER_HARDNEG.md](docs/EMBEDDER_HARDNEG.md)

## License

MIT for nanorag and both dependencies (public): [LICENSE](LICENSE).
