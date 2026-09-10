from fastmap._distances import Distance, InputError
from utils import shingler


def _weight_set(x):
    return {el: 1 for el in x}


def _to_set(value, shingle_size):
    if isinstance(value, str):
        return shingler(value, shingle_size=shingle_size)
    if isinstance(value, set):
        return value
    raise InputError("Jaccard distance needs strings, sets, or dictionaries")


def _to_weighted_set(value, shingle_size):
    if isinstance(value, dict):
        return value
    return _weight_set(_to_set(value, shingle_size))


def _match_inputs(x, y, shingle_size):
    if isinstance(x, dict) or isinstance(y, dict):
        return _to_weighted_set(x, shingle_size), _to_weighted_set(y, shingle_size)
    return _to_set(x, shingle_size), _to_set(y, shingle_size)


class Jaccard(Distance):
    def __init__(self, shingle_size=4):
        self._shingle_size = shingle_size

    @property
    def shingle_size(self):
        return self._shingle_size

    def _d(self, x, y):

        if isinstance(x, dict):
            return self._jac_dict(x, y)
        return self._jac_set(x, y)

    @staticmethod
    def get_name():
        return "Jaccard"

    def calculate(self, x, y) -> float:
        x, y = _match_inputs(x, y, self.shingle_size)
        d = self._d(x, y)
        return d

    def _jac_set(self, x, y):
        intersect = len(x.intersection(y))
        size1 = len(x)
        size2 = len(y)
        union = size1 + size2 - intersect
        return 0.0 if union == 0 else 1 - intersect / union

    def _jac_dict(self, x, y):
        keyset = x.keys() | y.keys()
        min_maxes = [
            (min(x.get(key, 0), y.get(key, 0)), max(x.get(key, 0), y.get(key, 0))) for key in keyset
        ]
        min_sum = 0
        max_sum = 0
        for mini, maxi in min_maxes:
            min_sum += mini
            max_sum += maxi
        return 0.0 if max_sum == 0 else 1 - min_sum / max_sum
