# Compatibility matrix

Pinned by **git submodule SHAs** in this repository (canonical for development).
Optional **package** versions apply when using `find_package` / installed prefixes.

| nanorag | tinyann | nanollm | Notes |
|---------|---------|---------|--------|
| 0.1.x   | ≥ 0.1.0 | ≥ 0.4.0 | Phase 0 seal + Phase 1 packaging |

## Formats

| Artifact | Version field | Notes |
|----------|---------------|--------|
| `meta.txt` | `nanorag_index_meta_version=2` | `n_real_chunks`, `has_no_evidence` |
| `embeddings.nctr` | format v1 (`NCTR`) | contrastive-v1 |
| `embeddings.nw2v` | format v1 (`NW2V`) | word2vec-v1 |
| `*.tann` | tinyann binary | host-endian |

## Development (default)

```bash
git clone --recurse-submodules https://github.com/mertugr/nanorag.git
cmake -S . -B build -DNANORAG_USE_SUBMODULES=ON
```

## Installed packages

```bash
# Install each lib (example prefix)
cmake -S third_party/tinyann -B build-tinyann -DTINYANN_BUILD_TESTS=OFF
cmake --build build-tinyann && cmake --install build-tinyann --prefix $PREFIX

cmake -S third_party/nanollm -B build-nanollm -DNANOLLM_BUILD_TESTS=OFF
cmake --build build-nanollm && cmake --install build-nanollm --prefix $PREFIX

cmake -S . -B build -DNANORAG_USE_SUBMODULES=OFF -DCMAKE_PREFIX_PATH=$PREFIX
cmake --build build && cmake --install build --prefix $PREFIX
```

`nanollm` may be private; package install still requires access to its sources or a private package feed.
