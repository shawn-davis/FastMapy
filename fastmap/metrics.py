"""Utilities for evaluating a FastMap embedding."""

import numpy as np


def pairwise_distances(objects, distance):
    """Return the symmetric pairwise-distance matrix for ``objects``."""
    objects = list(objects)
    matrix = np.zeros((len(objects), len(objects)), dtype=float)
    for left_index, left in enumerate(objects):
        for right_index in range(left_index + 1, len(objects)):
            value = distance.calculate(left, objects[right_index])
            matrix[left_index, right_index] = value
            matrix[right_index, left_index] = value
    return matrix


def normalized_stress(original_distances, embedding):
    """Return normalized Kruskal stress for an embedding."""
    original = _validate_distance_matrix(original_distances)
    embedded = _euclidean_distances(embedding)
    upper = np.triu_indices_from(original, k=1)
    denominator = np.sum(original[upper] ** 2)
    if denominator == 0:
        return 0.0
    return float(np.sqrt(np.sum((original[upper] - embedded[upper]) ** 2) / denominator))


def distance_correlation(original_distances, embedding, method="pearson"):
    """Correlate original and embedded pairwise distances.

    ``method='pearson'`` has no optional dependencies. ``method='spearman'``
    requires ``FastMapy[metrics]``.
    """
    original = _validate_distance_matrix(original_distances)
    embedded = _euclidean_distances(embedding)
    upper = np.triu_indices_from(original, k=1)
    left, right = original[upper], embedded[upper]
    if method == "pearson":
        if np.std(left) == 0 or np.std(right) == 0:
            return 1.0 if np.array_equal(left, right) else 0.0
        return float(np.corrcoef(left, right)[0, 1])
    if method == "spearman":
        try:
            from scipy.stats import spearmanr
        except ImportError as error:
            raise ImportError("Spearman correlation requires 'FastMapy[metrics]'.") from error
        return float(spearmanr(left, right).statistic)
    raise ValueError("method must be 'pearson' or 'spearman'")


def trustworthiness(original_distances, embedding, n_neighbors=5):
    """Return neighborhood trustworthiness from 0 to 1 without scikit-learn."""
    original = _validate_distance_matrix(original_distances)
    n_objects = len(original)
    if not isinstance(n_neighbors, int) or not 1 <= n_neighbors < n_objects / 2:
        raise ValueError("n_neighbors must be at least 1 and less than half the sample count")
    embedded = _euclidean_distances(embedding)
    original_order = np.argsort(original, axis=1)
    embedded_order = np.argsort(embedded, axis=1)
    ranks = np.empty_like(original_order)
    ranks[np.arange(n_objects)[:, None], original_order] = np.arange(n_objects)
    penalty = 0
    for index in range(n_objects):
        original_neighbors = set(original_order[index, 1 : n_neighbors + 1])
        for neighbor in embedded_order[index, 1 : n_neighbors + 1]:
            if neighbor not in original_neighbors:
                penalty += ranks[index, neighbor] - n_neighbors
    normalization = n_objects * n_neighbors * (2 * n_objects - 3 * n_neighbors - 1)
    return float(1 - (2 * penalty / normalization))


def _validate_distance_matrix(distances):
    matrix = np.asarray(distances, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("original_distances must be a square matrix")
    return matrix


def _euclidean_distances(embedding):
    points = np.asarray(embedding, dtype=float)
    if points.ndim != 2:
        raise ValueError("embedding must be a two-dimensional array")
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
