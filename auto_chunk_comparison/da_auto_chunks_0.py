# commit : 613401529c80cca77b44ced6cd59a7bc7b629f90

def factors(n):
    """ Return the factors of an integer
    https://stackoverflow.com/a/6800214/616616
    """
    seq = ([i, n // i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)
    return set(functools.reduce(list.__add__, seq))

def round_to(c, s):
    """ Return a chunk dimension that is close to an even multiple
    We want the largest factor of the dimension size (s) that is less than the
    desired chunk size, but not less than half, which is too much.  If no such
    factor exists then we just go with the original chunk size and accept an
    uneven chunk at the end.
    """
    try:
        return max(f for f in factors(s) if c / 2 <= f <= c)
    except ValueError:  # no matching factors within factor of two
        return max(1, int(c))

def auto_chunks(chunks, shape, limit, dtype, previous_chunks=None):
    """ Determine automatic chunks
    This takes in a chunks value that contains ``"auto"`` values in certain
    dimensions and replaces those values with concrete dimension sizes that try
    to get chunks to be of a certain size in bytes, provided by the ``limit=``
    keyword.  If multiple dimensions are marked as ``"auto"`` then they will
    all respond to meet the desired byte limit, trying to respect the aspect
    ratio of their dimensions in ``previous_chunks=``, if given.
    Parameters
    ----------
    chunks: Tuple
        A tuple of either dimensions or tuples of explicit chunk dimensions
        Some entries should be "auto"
    shape: Tuple[int]
    limit: int
        The maximum allowable size of a chunk in bytes
    previous_chunks: Tuple[Tuple[int]]
    See also
    --------
    normalize_chunks: for full docstring and parameters
    """
    chunks = list(chunks)

    autos = {i for i, c in enumerate(chunks) if c == 'auto'}
    if not autos:
        return tuple(chunks)

    if limit is None:
        limit = config.get('array.chunk-size')
        if isinstance(limit, str):
            limit = parse_bytes(limit)

    if dtype is None:
        raise TypeError("DType must be known for auto-chunking")

    if dtype.hasobject:
        raise NotImplementedError(
            "Can not use auto rechunking with object dtype. "
            "We are unable to estimate the size in bytes of object data")

    for x in tuple(chunks) + tuple(shape):
        if (isinstance(x, Number) and np.isnan(x) or
                isinstance(x, tuple) and np.isnan(x).any()):
            raise ValueError("Can not perform automatic rechunking with unknown "
                             "(nan) chunk sizes")

    limit = max(1, limit // dtype.itemsize)

    largest_block = np.prod([cs if isinstance(cs, Number) else max(cs)
                             for cs in chunks if cs != 'auto'])

    if previous_chunks:

        # Base ideal ratio on the median chunk size of the previous chunks
        result = {a: np.median(previous_chunks[a]) for a in autos}

        # How much larger or smaller the ideal chunk size is relative to what we have now
        multiplier = limit / largest_block / np.prod(list(result.values()))
        last_multiplier = 0
        last_autos = set()

        while (multiplier != last_multiplier or
               autos != last_autos):  # while things change
            last_multiplier = multiplier  # record previous values
            last_autos = set(autos)  # record previous values

            # Expand or contract each of the dimensions appropriately
            for a in sorted(autos):
                proposed = result[a] * multiplier ** (1 / len(autos))
                if proposed > shape[a]:  # we've hit the shape boundary
                    autos.remove(a)
                    largest_block *= shape[a]
                    chunks[a] = shape[a]
                    del result[a]
                else:
                    result[a] = round_to(proposed, shape[a])

            # recompute how much multiplier we have left, repeat
            multiplier = limit / largest_block / np.prod(list(result.values()))

        for k, v in result.items():
            chunks[k] = v
        return tuple(chunks)

    else:
        size = (limit / largest_block) ** (1 / len(autos))
        small = [i for i in autos if shape[i] < size]
        if small:
            for i in small:
                chunks[i] = (shape[i],)
            return auto_chunks(chunks, shape, limit, dtype)

        for i in autos:
            chunks[i] = round_to(size, shape[i])

        return tuple(chunks)
