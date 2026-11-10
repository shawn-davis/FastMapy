import math

import pytest

from fastmap import InputError
from fastmap.distances import L1, L2, Cosine, Jaccard, Lev


@pytest.mark.parametrize(
    ("metric", "left", "right", "expected"),
    [
        (L1(), [1, 2], [4, 6], 7),
        (L2(), [1, 2], [4, 6], 5),
        (L1(), {"a": 1, "b": 2}, {"a": 4, "c": 6}, 11),
        (L2(), {"a": 1, "b": 2}, {"a": 4, "c": 6}, 7),
        (Jaccard(shingle_size=1), "abc", "abd", 0.5),
        (Lev(), "fastmap", "fastmaps", 1),
    ],
)
def test_distance_calculations(metric, left, right, expected):
    assert metric.calculate(left, right) == pytest.approx(expected)


@pytest.mark.parametrize("metric", [L1(), L2()])
def test_vector_metrics_reject_scalar_inputs(metric):
    with pytest.raises(InputError):
        metric.calculate(1, 2)


def test_cosine_calculates_orthogonal_vector_distance():
    assert Cosine().calculate([1, 0], [0, 1]) == pytest.approx(math.sqrt(2))


def test_cosine_handles_zero_vectors():
    metric = Cosine()

    assert metric.calculate([0, 0], [0, 0]) == 0
    assert metric.calculate([0, 0], [1, 0]) == pytest.approx(math.sqrt(2))


def test_jaccard_handles_empty_and_mixed_weighted_inputs():
    metric = Jaccard(shingle_size=1)

    assert metric.calculate(set(), set()) == 0
    assert metric.calculate("ab", {"a": 1, "b": 1}) == 0


def test_jaccard_reuses_cached_string_shingles(monkeypatch):
    import fastmap.distances._jaccard as module

    module._string_shingles.cache_clear()
    calls = []
    original = module.shingler

    def counting_shingler(value, shingle_size):
        calls.append(value)
        return original(value, shingle_size)

    monkeypatch.setattr(module, "shingler", counting_shingler)
    metric = Jaccard(shingle_size=2)
    metric.calculate("abcd", "abce")
    metric.calculate("abce", "abcd")

    assert calls == ["abcd", "abce"]


def test_distance_names_are_public_and_stable():
    assert L1().get_name() == "L1"
    assert L2().get_name() == "L2"
    assert Cosine().get_name() == "Cosine"
    assert Jaccard().get_name() == "Jaccard"
    assert Lev().get_name() == "Levenshtein"
