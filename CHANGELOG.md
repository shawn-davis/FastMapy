# Changelog

## Unreleased

## 0.1.1 - 2026-09-11

### Changed

- Re-licensed FastMapy under the Apache License, Version 2.0.

### Added

- Modern Python packaging, local development dependencies, GitHub Actions CI, and an automated test suite.
- Input validation for FastMap construction and fitting.
- `FastMap.fit_many` for coordinated batches with distinct starting points and pivot-pair collision recovery.
- Optional `metrics`, `plots`, and `all` dependency groups for embedding evaluation and visualization.
- Embedding-quality metrics plus 2D/3D plotting and t-SNE/UMAP visualization helpers.
- Documentation for installation, supported metrics, threaded execution, and stochastic pivot selection.

### Changed

- Pivot selection now uses a valid random index while remaining non-deterministic by default.
- Degenerate zero-distance axes produce zero coordinates instead of failing during projection.
- Distance metrics consistently support dense sequences and sparse dictionaries where applicable.
- The threaded implementation reuses workers during pivot selection and transformation.

### Removed

- The unused Pebble dependency and the accidental Pandas runtime dependency.
