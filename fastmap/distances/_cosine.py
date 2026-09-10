from math import sqrt

from fastmap._distances import Distance, InputError
from fastmap.distances._helpers import _match_vec_inputs
from utils import is_list_like


def _norm(x):
    if isinstance(x, dict):
        x = x.values()
    return sqrt(sum([xi * xi for xi in x]))


def _dot(x, y):
    if isinstance(x, dict):
        all_keys = {*x}.union({*y})
        dot = sum([x.get(key, 0) * y.get(key, 0) for key in all_keys])
    else:
        dot = sum([xi * yi for (xi, yi) in zip(x, y)])
    return dot


def _d(x, y):

    norm_x = _norm(x)
    norm_y = _norm(y)

    dot = _dot(x, y)

    if norm_x == 0 and norm_y == 0:
        return 0.0
    if norm_x == 0 or norm_y == 0:
        return sqrt(2)

    sim = dot / (norm_x * norm_y)
    sim = min(max(sim, -1.0), 1.0)
    return sqrt(2 * (1 - sim))


class Cosine(Distance):
    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return "Cosine"

    def calculate(self, x, y) -> float:

        if not (
            (isinstance(x, dict) or is_list_like(x)) and (isinstance(y, dict) or is_list_like(y))
        ):
            raise InputError("Cosine distance needs to be non-set, list like objects")

        x, y = _match_vec_inputs(x, y)
        d = _d(x, y)

        return d
