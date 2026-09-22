import pickle

import numpy as np
import pytest

from fastmap import FastMap
from fastmap._fastmap import _MODEL_FORMAT, _MODEL_FORMAT_VERSION, ModelError
from fastmap.distances import L2, Jaccard


def test_save_and_load_preserves_a_fitted_model(monkeypatch, tmp_path):
    vectors = [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]
    monkeypatch.setattr("fastmap._fastmap.random.randrange", lambda count: 0)
    model = FastMap(dim=1, distance=L2, iters=2).fit(vectors)
    path = tmp_path / "model.fastmap"

    assert model.save(path) is model
    loaded = FastMap.load(path)

    assert loaded.dim == model.dim
    assert loaded.distance == model.distance
    assert np.allclose(loaded.transform(vectors), model.transform(vectors))


def test_save_and_load_preserves_distance_configuration(monkeypatch, tmp_path):
    strings = ["abcd", "abce", "xyz"]
    monkeypatch.setattr("fastmap._fastmap.random.randrange", lambda count: 0)
    model = FastMap(dim=1, distance=Jaccard, dist_args={"shingle_size": 2}).fit(strings)
    path = tmp_path / "jaccard.fastmap"

    loaded = model.save(path).load(path)

    assert loaded._distance.shingle_size == 2
    assert np.allclose(loaded.transform(strings), model.transform(strings))


def test_save_rejects_an_unfitted_model(tmp_path):
    with pytest.raises(ModelError, match="fully fitted"):
        FastMap(dim=1, distance=L2).save(tmp_path / "model.fastmap")


def test_load_rejects_non_fastmapy_data(tmp_path):
    path = tmp_path / "not-a-model.fastmap"
    with path.open("wb") as model_file:
        pickle.dump({"format": "something-else"}, model_file)

    with pytest.raises(ModelError, match="not a FastMapy model"):
        FastMap.load(path)


def test_load_rejects_malformed_fitted_model(monkeypatch, tmp_path):
    vectors = [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]
    monkeypatch.setattr("fastmap._fastmap.random.randrange", lambda count: 0)
    model = FastMap(dim=1, distance=L2, iters=2).fit(vectors)
    model._pivots[0] = object()
    path = tmp_path / "malformed.fastmap"
    with path.open("wb") as model_file:
        pickle.dump(
            {
                "format": _MODEL_FORMAT,
                "format_version": _MODEL_FORMAT_VERSION,
                "model": model,
            },
            model_file,
        )

    with pytest.raises(ModelError, match="invalid pivot"):
        FastMap.load(path)
