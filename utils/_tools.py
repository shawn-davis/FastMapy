from collections import abc

import numpy as np


def gen_flatten(iterables):
    """
    General-purpose collection flattener for objects from second-level iterables.
    Parameters
    ----------
    iterables: The iterable of iterables to flatten

    Returns
    -------
    List
    """
    flattened = (elem for iterable in iterables for elem in iterable)
    return list(flattened)


def shingler(s, shingle_size: int) -> set[str]:
    """
    Shingles the string of the input by the parameterized shingle-size.
    Parameters
    ----------
    s: The string (or string-like) object to shingle
    shingle_size: The size of the character shingles

    Returns
    -------
    Set[str]: The set of shingled character strings
    """
    input_string = str(s)
    if shingle_size >= len(input_string):
        return set(input_string)
    return set(
        [input_string[i : i + shingle_size] for i in range(len(input_string) - shingle_size + 1)]
    )


def is_list_like(obj) -> bool:
    """
    Adapted from Pandas is_list_like; excludes dictionaries and sets.

    Parameters
    ----------
    obj: The object to test for list-likeness

    Returns
    -------
    bool
    """
    return (
        isinstance(obj, abc.Iterable)
        and not isinstance(obj, (str, bytes, dict, set))
        and not (isinstance(obj, np.ndarray) and obj.ndim == 0)
    )
