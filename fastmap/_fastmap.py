from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor as Executor
import multiprocessing
import numpy as np
from math import sqrt
from dataclasses import dataclass
from typing import List
import random
from pandas.api.types import is_list_like

class ModelError(Exception):

    def __init__(self, message):
        self.message = message

@dataclass
class Pivots:
    left: object
    right: object
    distance: float


class FastMap:

    def __init__(self, dim, distance, iters=5, cores=1):
        self._dim = dim
        self._distance = distance
        self._iters = iters
        self._cores = cores
        self._pivots: List[Pivots]  = []#[None for _ in range(dim)]

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
        if new_cores > 0 and isinstance(new_cores, int):
            self._cores = min(new_cores, multiprocessing.cpu_count())
        else:
            print("Please enter an int greater than 0")

    def _compute_proj_i(self, index, piv_dist, left, right, obj, obj_proj):
        left_dist = self.dist(left, left.proj, obj, obj_proj, index)
        right_dist = self.dist(right, right.proj, obj, obj_proj, index)
        numer = pow(left_dist, 2) + pow(piv_dist, 2) - pow(right_dist, 2)
        denom = 2 * piv_dist
        return numer / denom

    def _i_proj(self, obj, index):
        assert len(self.pivots) >= index
        x_proj = np.zeros(self._dim)
        for i in range(0, index):
            left = self._pivots[i].left
            right = self._pivots[i].right
            piv_dist = self._pivots[i].dist
            x_proj[i] = self._compute_proj_i(index, piv_dist, left, right, obj, x_proj)
        return x_proj

    def _dist(self, x, x_proj, y, y_proj, index):
        d_sq = pow(self._distance(x, y), 2)
        diff_sq = sum([pow(x_i - y_i, 2) for (x_i, y_i) in zip(x_proj[0:index], y_proj[0:index])])
        return sqrt(max(d_sq - diff_sq, 0))

    def fit(self, X):
        self._pivots: List[Pivots] = []
        if self._cores == 1:
            self._serial_pivot_finder(X)
        else:
            self._parallel_pivot_finder(X)
        return self

    def transform(self, X):
        if len(self._pivots) != self._dim:
            raise ModelError("Model not built or deficient")
        if is_list_like(X):
            return [self._i_proj(x, self._dim) for x in X]
        else:
            return self._i_proj(X, self._dim)

    def fit_transform(self, X):
        self._pivots: List[Pivots] = []
        self.fit(X).transform(X)

    def _serial_pivot_finder(self, X):

        N = len(X)
        for k in range(self._dim):
            print('Working on ' + str(k) + 'D projection')
            left_pivot_index = random.randint(0, N)
            right_pivot_index = left_pivot_index
            for m in range(self._iters):
                left_pivot_index = right_pivot_index
                left_pivot = X[left_pivot_index]
                max_dist = 0.0
                for i in range(N):
                    right_candidate = X[i]
                    left_proj = self._i_proj(left_pivot, k)
                    right_proj = self._i_proj(right_candidate, k)
                    d = self._dist(left_pivot, left_proj, right_candidate, right_proj, k)
                    if d > max_dist:
                        right_pivot_index = i
                        max_dist = d
            print('Left Pivot: ' + str(left_pivot_index) + '\nRight Pivot: ' + str(right_pivot_index))
            left = X[left_pivot_index]
            right = X[right_pivot_index]
            final_pivots = Pivots(left, right, max_dist)
            self._pivots.insert(k, final_pivots)

    def _parallel_pivot_finder(self, X):

        N = len(X)
        for k in range(self._dim):
            print('Working on ' + str(k) + 'D projection')
            left_pivot_index = random.randint(0, N)
            right_pivot_index = left_pivot_index
            for m in range(self._iters):
                left_pivot_index = right_pivot_index
                left_pivot = X[left_pivot_index]
                with Executor(max_workers=self._cores) as executor:
                    map_results = executor.map(self._mapper, [(X[i], i, left_pivot, k) for i in range(N)])

                    distributor = defaultdict(list)
                    for key, value in map_results:
                        distributor[key].append(value)

                    reduced = executor.map(self._reducer, distributor.items())
                    reduced.sort(key=lambda x: -x[1])
                right_pivot_index, max_dist = reduced[0]
            print('Left Pivot: ' + str(left_pivot_index) + '\nRight Pivot: ' + str(right_pivot_index))
            left = X[left_pivot_index]
            right = X[right_pivot_index]
            final_pivots = Pivots(left, right, max_dist)
            self._pivots.insert(k, final_pivots)



    def _mapper(self, entry):
        right_candidate, i, left_pivot, k = entry
        left_proj = self._i_proj(left_pivot, k)
        right_proj = self._i_proj(right_candidate, k)
        d = self._dist(left_pivot, left_proj, right_candidate, right_proj, k)
        return i % (self._cores * 4), (i, d)

    def _reducer(self, entry):
        (_, distances) = entry
        distances.sort(key=lambda x: -x[1])
        return distances[0]