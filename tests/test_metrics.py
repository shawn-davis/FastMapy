import numpy as np
import pytest

from fastmap.distances import L2
from fastmap.metrics import (
    distance_correlation,
    normalized_stress,
    pairwise_distances,
    trustworthiness,
)
from fastmap.plots import plot_embedding, reduce_for_plot


@pytest.fixture
def original_distances():
    return np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])


@pytest.fixture
def embedding():
    return np.array([[0.0], [1.0], [2.0]])


def test_pairwise_distances_uses_the_supplied_metric():
    expected = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    assert np.array_equal(pairwise_distances([[0], [1], [2]], L2()), expected)


def test_embedding_metrics_are_perfect_for_exact_distances(original_distances, embedding):
    assert normalized_stress(original_distances, embedding) == 0
    assert distance_correlation(original_distances, embedding) == 1
    assert trustworthiness(original_distances, embedding, n_neighbors=1) == 1


def test_plot_helpers_validate_arguments_before_loading_optional_dependencies(embedding):
    with pytest.raises(ValueError, match="tsne"):
        reduce_for_plot(embedding, method="pca")
    with pytest.raises(ValueError, match="match"):
        plot_embedding(embedding, dimensions=2, show=False)
