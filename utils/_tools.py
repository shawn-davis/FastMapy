def gen_flatten(iterables):
    flattened = (elem for iterable in iterables for elem in iterable)
    return list(flattened)


def shingler(s, shingle_size):
    input_string = str(s)
    if shingle_size >= len(input_string):
        return set(input_string)
    return set([input_string[i:i+shingle_size] for i in range(len(input_string) - shingle_size + 1)])