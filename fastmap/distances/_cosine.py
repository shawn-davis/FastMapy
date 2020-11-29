from fastmap import Distance, InputError
from math import sqrt
from pandas.api.types import is_list_like


def _to_dict(x):
    if isinstance(x, dict):
        return x
    else:
        return {idx: count for (idx, count) in enumerate(x) if count != 0}


def _match_inputs(x, y):
    if type(x) == type(y):
        return x, y
    elif isinstance(x, dict) or isinstance(y, dict):
        d, coll = (x, y) if isinstance(x, dict) else (y, x)
        return (d, _to_dict(coll))
    else:
        assert len(x) == len(y), "Non-set, non-dict list-like inputs must have the same length"
        return x, y


class Cosine(Distance):

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return 'Cosine'

    def distance(self, x, y):
        if not is_list_like(x, allow_sets=False) or not is_list_like(y, allow_sets=False):
            raise InputError("Cosine distance needs to be non-set, list like objects")
        norm_x = sqrt(sum([xi * xi for xi in list(x.get_item().values())]))
        norm_y = sqrt(sum([yi * yi for yi in list(y.get_item().values())]))
        all_keys = {*x.get_item()}.union({*y.get_item()})
        dot = sum([x.get_item().get(key, 0) * y.get_item().get(key, 0) for key in all_keys])
        sim = min(dot / (norm_x * norm_y), 1.0)
        return sqrt(2 * (1 - sim))