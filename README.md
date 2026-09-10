# FastMapy

FastMapy is a Python implementation of the FastMap<sup id="a1">[1](#f1)</sup> multidimensional-scaling technique. It
embeds objects into a vector space from a supplied distance metric, attempting to preserve their relative distances.

This package has common distance metrics already defined and ready to use over appropriate objects, such as
Jaccard distance over character shingled _n_-gram strings or Levenshtein edit distance for embedding string objects.
Euclidean distance and taxi cab distance are also available for vector objects. Dictionary objects also work assuming a
sparse vector style dictionary of _{index: count}_ where index can be an actual vector index or a token and its
occurrence count.

Threaded execution can be enabled for model building and object transformation with the `cores` argument. It is set to
serial execution (`cores=1`) by default. The benefit depends on the distance metric and runtime.

## Installation

```bash
python -m pip install FastMapy
```

For local development:

```bash
python -m pip install -e '.[dev]'
pytest
```

Optional features can be installed individually with `FastMapy[metrics]` or `FastMapy[plots]`, or together with
`FastMapy[all]`. t-SNE and UMAP are included only in the plotting extra.

## Usage

```
from fastmap.distances import Jaccard
import fastmap

fm_model = fastmap.FastMap(dim=8, distance=Jaccard, dist_args={'shingle_size':4})

embedding = fm_model.fit_transform(string_data)
```
The target vector space is eight-dimensional and strings are shingled into four-grams before their distances are
computed. `fit_transform` returns one NumPy array per input object.

`fit` requires more training objects than requested dimensions. `transform` expects a collection of objects; wrap a
single dense vector in a one-element collection, such as `model.transform([[1.0, 2.0]])`.

### Metrics

`fastmap.metrics` provides a pairwise-distance helper plus normalized stress, Pearson/Spearman distance correlation,
and trustworthiness. Pass the original pairwise-distance matrix and the resulting embedding to the evaluators. Spearman
correlation requires the `metrics` extra.

```python
from fastmap.metrics import distance_correlation, pairwise_distances, trustworthiness

original_distances = pairwise_distances(string_data, Jaccard(shingle_size=4))
print(distance_correlation(original_distances, embedding))
print(trustworthiness(original_distances, embedding, n_neighbors=5))
```

### Plots

`fastmap.plots.plot_embedding` renders 2D or 3D embeddings. For embeddings with more dimensions,
`reduce_for_plot` performs a visualization-only t-SNE or UMAP reduction; it does not train or stack another FastMap
model. These helpers require the `plots` extra.

```python
from fastmap.plots import plot_embedding, reduce_for_plot

plot_embedding(embedding_2d, dimensions=2)
plot_embedding(embedding_3d, dimensions=3)

umap_2d = reduce_for_plot(embedding, method="umap", n_components=2)
plot_embedding(umap_2d, dimensions=2)
```

### Reproducibility

FastMap selects an initial pivot randomly for each dimension. Consequently, unseeded fits are intentionally
non-deterministic: two fits over identical data can produce different, valid embeddings. Tests and experiments that need
repeatability should control Python's random-number generator before fitting.

### Built-in metrics

| Metric | Inputs |
| --- | --- |
| `L1` | Dense sequences or sparse `{index: value}` dictionaries |
| `L2` | Dense sequences or sparse `{index: value}` dictionaries |
| `Cosine` | Dense sequences or sparse dictionaries; returns chord distance |
| `Jaccard` | Strings, sets, or weighted dictionaries |
| `Lev` | Strings and sequence-like objects |

`cores` enables threaded fitting and transformation. It defaults to `1`; any speedup depends on the distance metric and
runtime.

### Fitting a batch of distinct models

Use `FastMap.fit_many` to fit several models with identical settings against one training collection:

```python
models = fastmap.FastMap.fit_many(
    string_data,
    count=4,
    dim=8,
    distance=Jaccard,
    dist_args={"shingle_size": 4},
)
```

Each model starts every dimension from a distinct training-object index. The batch also avoids reusing an unordered
pivot pair anywhere in the batch. If a pair collides, FastMap retries that dimension with another unused starting point
and retains all prior dimensions. When no distinct pair can be found within `pair_retries` attempts, the collision is
retained and reported by the model's `pivot_pair_collisions` property. `count` cannot exceed the number of training
objects.

## References
<b id="f1">1</b> Proceedings of the 1995 ACM SIGMOD international conference on Management of data  - SIGMOD  ’95. (1995). doi:10.1145/223784 [↩](#a1)
