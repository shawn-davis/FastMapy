from fastmap import Distance, InputError
from math import sqrt, log, exp
from utils import is_list_like
from fastmap.distances._helpers import _match_vec_inputs


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

    log_sim = log(dot) - log(norm_x) - log(norm_y)
    sim = exp(log_sim)
    return sqrt(2 * (1 - sim))


class Cosine(Distance):

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return 'Cosine'

    def calculate(self, x, y) -> float:

        if not (is_list_like(x, allow_sets=False) and is_list_like(y, allow_sets=False)):
            raise InputError("Cosine distance needs to be non-set, list like objects")

        x, y = _match_vec_inputs(x, y)
        d = _d(x, y)

        return d
