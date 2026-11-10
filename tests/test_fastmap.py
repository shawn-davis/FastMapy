import numpy as np
import pytest

from fastmap import FastMap
from fastmap._fastmap import ModelError
from fastmap.distances import L2


@pytest.fixture
def vectors():
    return [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]


def test_fit_transform_creates_finite_vectors(monkeypatch, vectors):
    monkeypatch.setattr("fastmap._fastmap.random.randrange", lambda count: 0)
    model = FastMap(dim=1, distance=L2, iters=2)

    embedding = model.fit_transform(vectors)

    assert len(embedding) == len(vectors)
    assert all(vector.shape == (1,) for vector in embedding)
    assert np.isfinite(embedding).all()
    assert model.dim == 1
    assert model.distance == "L2"


def test_fit_transform_accepts_an_iterator(monkeypatch, vectors):
    monkeypatch.setattr("fastmap._fastmap.random.randrange", lambda count: 0)

    embedding = FastMap(dim=1, distance=L2, iters=2).fit_transform(iter(vectors))

    assert len(embedding) == len(vectors)


def test_transform_accepts_a_collection(monkeypatch, vectors):
    monkeypatch.setattr("fastmap._fastmap.random.randrange", lambda count: 0)
    model = FastMap(dim=1, distance=L2, iters=2).fit(vectors)

    collection_embedding = model.transform(vectors[:2])

    assert len(collection_embedding) == 2
    assert all(vector.shape == (1,) for vector in collection_embedding)


def test_parallel_transform_matches_serial_transform(monkeypatch, vectors):
    monkeypatch.setattr("fastmap._fastmap.random.randrange", lambda count: 0)
    serial_model = FastMap(dim=1, distance=L2, iters=2, cores=1).fit(vectors)
    parallel_model = FastMap(dim=1, distance=L2, iters=2, cores=2).fit(vectors)

    assert np.allclose(serial_model.transform(vectors), parallel_model.transform(vectors))


def test_transform_requires_a_fitted_model(vectors):
    with pytest.raises(ModelError, match="Model not built or deficient"):
        FastMap(dim=1, distance=L2).transform(vectors)


def test_fit_rejects_empty_training_data():
    with pytest.raises(ValueError, match="more objects"):
        FastMap(dim=1, distance=L2).fit([])


def test_fit_handles_identical_objects():
    embedding = FastMap(dim=1, distance=L2).fit_transform([[1, 1], [1, 1]])
    assert np.allclose(embedding, 0)


def test_distance_cache_reuses_symmetric_metric_evaluations():
    calls = []

    class CountingDistance(L2):
        def calculate(self, left, right):
            calls.append((id(left), id(right)))
            return super().calculate(left, right)

    vectors = [[0.0], [1.0], [2.0]]
    model = FastMap(dim=1, distance=CountingDistance, iters=1, cache_distances=True)
    model.fit(vectors)
    model._metric_distance(vectors[0], vectors[1])
    before_reverse = len(calls)
    model._metric_distance(vectors[1], vectors[0])

    assert len(calls) == before_reverse
    assert model.cache_distances


@pytest.mark.parametrize("dim", [0, -1, 1.5, True])
def test_dimension_must_be_a_positive_integer(dim):
    with pytest.raises(ValueError, match="positive integer"):
        FastMap(dim=dim, distance=L2)


def test_fit_requires_more_objects_than_dimensions(vectors):
    with pytest.raises(ValueError, match="more objects"):
        FastMap(dim=len(vectors), distance=L2).fit(vectors)


@pytest.mark.parametrize("cores", [0, -1, 1.5, True])
def test_cores_must_be_a_positive_integer(cores):
    with pytest.raises(ValueError, match="positive integer"):
        FastMap(dim=1, distance=L2, cores=cores)


def test_fit_many_uses_distinct_starting_points_and_pairs(monkeypatch, vectors):
    monkeypatch.setattr("fastmap._fastmap.random.choice", lambda choices: choices[0])
    vectors.append([9.0, 12.0])

    models = FastMap.fit_many(vectors, count=3, dim=1, distance=L2, iters=1)

    pairs = [
        frozenset((model._pivots[0].left_index, model._pivots[0].right_index)) for model in models
    ]
    starts = [model._pivots[0].left_index for model in models]
    assert starts == [0, 1, 2]
    assert len(set(pairs)) == len(models)
    assert all(not model.pivot_pair_collisions for model in models)


def test_fit_many_records_exhausted_pair_collisions(monkeypatch, vectors):
    monkeypatch.setattr("fastmap._fastmap.random.choice", lambda choices: choices[0])

    models = FastMap.fit_many(vectors, count=2, dim=1, distance=L2, iters=2, pair_retries=2)

    assert models[0].pivot_pair_collisions == ()
    assert models[1].pivot_pair_collisions == (0,)


def test_fit_many_rejects_more_models_than_training_objects(vectors):
    with pytest.raises(ValueError, match="cannot exceed"):
        FastMap.fit_many(vectors, count=4, dim=1, distance=L2)
