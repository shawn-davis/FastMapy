from fastmap._distances import Distance, InputError
from fastmap.distances._helpers import _match_vec_inputs
from utils import is_list_like


def _d(x, y):
    if isinstance(x, dict):
        all_keys = {*x}.union({*y})
        diffs = [x.get(key, 0) - y.get(key, 0) for key in all_keys]
    else:
        diffs = [xi - yi for (xi, yi) in zip(x, y)]
    return sum([abs(diff) for diff in diffs])


class L1(Distance):
    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return "L1"

    def calculate(self, x, y) -> float:

        if not (
            (isinstance(x, dict) or is_list_like(x)) and (isinstance(y, dict) or is_list_like(y))
        ):
            raise InputError("L1 distance needs non-set, list-like objects")

        x, y = _match_vec_inputs(x, y)
        d = _d(x, y)

        return d
