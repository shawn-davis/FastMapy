import multiprocessing
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor as Executor
from dataclasses import dataclass
from math import sqrt
from numbers import Real
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from utils import is_list_like


class ModelError(Exception):
    """Raised when a FastMap model cannot be fitted or used."""


_MODEL_FORMAT = "fastmapy.model"
_MODEL_FORMAT_VERSION = 1


@dataclass
class Pivots:
    left: object
    left_index: int
    left_proj: np.ndarray
    right: object
    right_index: int
    right_proj: np.ndarray
    distance: float


class FastMap:
    def __init__(self, dim, distance, dist_args=None, obj_transformer=None, iters=5, cores=1):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not callable(distance):
            raise TypeError("distance must be a callable distance-metric class")
        if dist_args is not None and not isinstance(dist_args, dict):
            raise TypeError("dist_args must be a dictionary or None")
        if obj_transformer is not None and not callable(obj_transformer):
            raise TypeError("obj_transformer must be callable or None")
        if not isinstance(iters, int) or isinstance(iters, bool) or iters < 1:
            raise ValueError("iters must be a positive integer")

        self._dim = dim
        self._distance = distance(**(dist_args or {}))
        self._obj_transformer = obj_transformer
        self._iters = iters
        self._pivots: list[Pivots] = []
        self._pivot_pair_collisions: list[int] = []
        self.cores = cores

    @classmethod
    def fit_many(
        cls,
        X,
        count,
        dim,
        distance,
        dist_args=None,
        obj_transformer=None,
        iters=5,
        cores=1,
        pair_retries=10,
    ):
        """Fit several distinct models while avoiding repeated pivot pairs.

        Models receive distinct starting indexes at every dimension. If a candidate pair
        matches a pair already reserved by an earlier model, only that dimension is retried.
        A collision is retained after ``pair_retries`` attempts when no distinct pair is found.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer")
        if not isinstance(pair_retries, int) or isinstance(pair_retries, bool) or pair_retries < 1:
            raise ValueError("pair_retries must be a positive integer")
        try:
            X = list(X)
        except TypeError as error:
            raise TypeError("X must be an iterable of objects") from error
        if count > len(X):
            raise ValueError("count cannot exceed the number of training objects")

        models = [
            cls(dim, distance, dist_args, obj_transformer, iters, cores) for _ in range(count)
        ]
        for model in models:
            model._validate_training_data(X)

        reserved_pairs = set()
        used_starts = [set() for _ in range(dim)]
        for model in models:
            model._fit_with_reservations(X, reserved_pairs, used_starts, pair_retries)
        return models

    @property
    def dim(self):
        return self._dim

    @property
    def distance(self):
        return self._distance.get_name()

    @property
    def iters(self):
        return self._iters

    @property
    def cores(self):
        return self._cores

    @cores.setter
    def cores(self, new_cores):
        if isinstance(new_cores, int) and not isinstance(new_cores, bool) and new_cores > 0:
            self._cores = min(new_cores, multiprocessing.cpu_count())
        else:
            raise ValueError("cores must be a positive integer")

    @property
    def pivot_pair_collisions(self):
        """Dimensions where batch fitting exhausted its distinct-pair retries."""
        return tuple(self._pivot_pair_collisions)

    def save(self, path):
        """Persist a fitted model to ``path``.

        The model is stored with Python pickle in a versioned FastMapy envelope. Only
        load files from trusted sources. Custom distance classes and object transformers
        must be importable module-level objects when the model is loaded.
        """
        if len(self._pivots) != self._dim:
            raise ModelError("Only a fully fitted model can be saved")

        destination = Path(path)
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(f"Model path is a directory: {destination}")

        temporary_path = None
        try:
            with NamedTemporaryFile("wb", dir=destination.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                pickle.dump(
                    {
                        "format": _MODEL_FORMAT,
                        "format_version": _MODEL_FORMAT_VERSION,
                        "model": self,
                    },
                    temporary,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
        except (OSError, pickle.PickleError, TypeError, AttributeError) as error:
            raise ModelError(f"Unable to save model to {destination}") from error
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return self

    @classmethod
    def load(cls, path):
        """Load a model saved with :meth:`save`.

        Pickle loading can execute code, so ``path`` must refer to a model file from a
        trusted source. The persisted format version is checked before returning it.
        """
        source = Path(path)
        try:
            with source.open("rb") as model_file:
                payload = pickle.load(model_file)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError) as error:
            raise ModelError(f"Unable to load model from {source}") from error

        if not isinstance(payload, dict) or payload.get("format") != _MODEL_FORMAT:
            raise ModelError("File is not a FastMapy model")
        if payload.get("format_version") != _MODEL_FORMAT_VERSION:
            raise ModelError("Unsupported FastMapy model format version")

        model = payload.get("model")
        if not isinstance(model, cls):
            raise ModelError("Persisted model has an invalid type")
        cls._validate_loaded_model(model)
        return model

    @staticmethod
    def _validate_loaded_model(model):
        """Reject persisted state that cannot safely support transformation."""
        if not isinstance(getattr(model, "_dim", None), int) or isinstance(model._dim, bool):
            raise ModelError("Persisted model has an invalid dimension")
        if model._dim < 1:
            raise ModelError("Persisted model has an invalid dimension")
        if not isinstance(getattr(model, "_iters", None), int) or model._iters < 1:
            raise ModelError("Persisted model has invalid iteration settings")
        if not isinstance(getattr(model, "_cores", None), int) or model._cores < 1:
            raise ModelError("Persisted model has invalid core settings")
        if not callable(getattr(getattr(model, "_distance", None), "calculate", None)):
            raise ModelError("Persisted model has an invalid distance metric")
        transformer = getattr(model, "_obj_transformer", None)
        if transformer is not None and not callable(transformer):
            raise ModelError("Persisted model has an invalid object transformer")
        if not isinstance(getattr(model, "_pivots", None), list):
            raise ModelError("Persisted model is not fully fitted")
        if len(model._pivots) != model._dim:
            raise ModelError("Persisted model is not fully fitted")
        if not isinstance(getattr(model, "_pivot_pair_collisions", None), list):
            raise ModelError("Persisted model has invalid collision data")

        for pivot in model._pivots:
            if not isinstance(pivot, Pivots):
                raise ModelError("Persisted model has an invalid pivot")
            if not all(
                isinstance(index, int) and not isinstance(index, bool) and index >= 0
                for index in (pivot.left_index, pivot.right_index)
            ):
                raise ModelError("Persisted model has invalid pivot indexes")
            if not isinstance(pivot.distance, Real):
                raise ModelError("Persisted model has an invalid pivot distance")
            if not all(
                isinstance(projection, np.ndarray) and projection.shape == (model._dim,)
                for projection in (pivot.left_proj, pivot.right_proj)
            ):
                raise ModelError("Persisted model has invalid pivot projections")

        if not all(
            isinstance(index, int) and not isinstance(index, bool) and 0 <= index < model._dim
            for index in model._pivot_pair_collisions
        ):
            raise ModelError("Persisted model has invalid collision data")

    def _compute_proj_i(self, index, pivots, obj, obj_proj):
        if pivots.distance == 0:
            return 0.0

        left_dist = self._dist(pivots.left, pivots.left_proj, obj, obj_proj, index)
        right_dist = self._dist(pivots.right, pivots.right_proj, obj, obj_proj, index)
        numer = pow(left_dist, 2) + pow(pivots.distance, 2) - pow(right_dist, 2)
        denom = 2 * pivots.distance
        return numer / denom

    def _i_proj(self, obj, index):
        assert len(self._pivots) >= index
        x_proj = np.zeros(self._dim)
        for i in range(0, index):
            x_proj[i] = self._compute_proj_i(index, self._pivots[i], obj, x_proj)
        return x_proj

    def _dist(self, x, x_proj, y, y_proj, index):
        d_sq = pow(self._distance.calculate(x, y), 2)
        diff_sq = sum([pow(x_i - y_i, 2) for (x_i, y_i) in zip(x_proj[0:index], y_proj[0:index])])
        return sqrt(max(d_sq - diff_sq, 0))

    def fit(self, X):
        try:
            X = list(X)
        except TypeError as error:
            raise TypeError("X must be an iterable of objects") from error

        X = self._prepare_training_data(X)
        if self._cores == 1:
            self._serial_pivot_finder(X)
        else:
            self._parallel_pivot_finder(X)
        return self

    def transform(self, X):
        if len(self._pivots) != self._dim:
            raise ModelError("Model not built or deficient")
        if is_list_like(X):
            if self._obj_transformer is not None:
                X = [self._obj_transformer(x) for x in X]
            if self._cores == 1:
                return [self._i_proj(x, self._dim) for x in X]
            else:
                return self._parallel_transform(X)
        else:
            if self._obj_transformer is not None:
                X = self._obj_transformer(X)
            return self._i_proj(X, self._dim)

    def _parallel_transform(self, X):
        with Executor(max_workers=self._cores) as executor:
            return list(executor.map(lambda obj: self._i_proj(obj, self._dim), X))

    def fit_transform(self, X):
        X = list(X)
        return self.fit(X).transform(X)

    def _validate_training_data(self, X):
        if len(X) <= self._dim:
            raise ValueError("X must contain more objects than the requested dimensions")

    def _prepare_training_data(self, X):
        self._validate_training_data(X)
        if self._obj_transformer is not None:
            X = [self._obj_transformer(x) for x in X]
        self._pivots = []
        self._pivot_pair_collisions = []
        return X

    def _fit_with_reservations(self, X, reserved_pairs, used_starts, pair_retries):
        X = self._prepare_training_data(X)
        N = len(X)
        executor = Executor(max_workers=self._cores) if self._cores > 1 else None
        try:
            for k in range(self._dim):
                selected = None
                last_result = None
                last_start = None
                attempted_starts = set()
                for _ in range(pair_retries):
                    start_index = self._batch_start_index(N, used_starts[k], attempted_starts)
                    if start_index is None:
                        break
                    attempted_starts.add(start_index)
                    last_start = start_index
                    result = self._find_pivot_pair(X, k, start_index, executor)
                    last_result = result
                    pair = frozenset(result[:2])
                    if pair not in reserved_pairs:
                        selected = result
                        used_starts[k].add(start_index)
                        reserved_pairs.add(pair)
                        break
                if selected is None:
                    selected = last_result
                    used_starts[k].add(last_start)
                    self._pivot_pair_collisions.append(k)
                self._append_pivots(X, k, *selected)
        finally:
            if executor is not None:
                executor.shutdown()
        return self

    @staticmethod
    def _batch_start_index(N, used_starts, attempted_starts):
        available = sorted(set(range(N)) - used_starts - attempted_starts)
        if not available:
            return None
        return random.choice(available)

    def _serial_pivot_finder(self, X):
        N = len(X)
        for k in range(self._dim):
            result = self._find_pivot_pair(X, k, random.randrange(N))
            self._append_pivots(X, k, *result)

    def _parallel_pivot_finder(self, X):
        N = len(X)
        with Executor(max_workers=self._cores) as executor:
            for k in range(self._dim):
                result = self._find_pivot_pair(X, k, random.randrange(N), executor)
                self._append_pivots(X, k, *result)

    def _find_pivot_pair(self, X, k, start_index, executor=None):
        left_pivot_index = start_index
        right_pivot_index = start_index
        max_dist = 0.0
        for _ in range(self._iters):
            left_pivot_index = right_pivot_index
            left_pivot = X[left_pivot_index]
            left_proj = self._i_proj(left_pivot, k)
            if executor is None:
                distances = (
                    self._distance_to_pivot((candidate, index, left_pivot, left_proj, k))
                    for index, candidate in enumerate(X)
                )
            else:
                distances = executor.map(
                    self._distance_to_pivot,
                    [
                        (candidate, index, left_pivot, left_proj, k)
                        for index, candidate in enumerate(X)
                    ],
                )
            right_pivot_index, max_dist = max(distances, key=lambda item: item[1])
        return left_pivot_index, right_pivot_index, max_dist

    def _append_pivots(self, X, k, left_pivot_index, right_pivot_index, max_dist):
        left = X[left_pivot_index]
        left_proj = self._i_proj(left, k)
        right = X[right_pivot_index]
        right_proj = self._i_proj(right, k)
        self._pivots.insert(
            k,
            Pivots(
                left, left_pivot_index, left_proj, right, right_pivot_index, right_proj, max_dist
            ),
        )

    def _distance_to_pivot(self, entry):
        right_candidate, index, left_pivot, left_proj, k = entry
        right_proj = self._i_proj(right_candidate, k)
        d = self._dist(left_pivot, left_proj, right_candidate, right_proj, k)
        return index, d
