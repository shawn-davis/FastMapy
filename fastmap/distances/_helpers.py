def _to_dict(x):
    if isinstance(x, dict):
        return x
    else:
        return {idx: count for (idx, count) in enumerate(x) if count != 0}


def _match_vec_inputs(x, y):
    """Normalize dense and sparse vector pairs to compatible representations."""
    if isinstance(x, dict) and isinstance(y, dict):
        return x, y
    if isinstance(x, dict):
        return x, _to_dict(y)
    if isinstance(y, dict):
        return _to_dict(x), y
    if len(x) != len(y):
        raise ValueError("Dense vectors must have the same length")
    return x, y
