"profiler: profile function and save the results"

import cProfile
import pstats
from contextlib import contextmanager
from functools import wraps
from io import IOBase
from pathlib import Path
from sys import stdout
from typing import Optional, Union


@contextmanager
def _dummy_cm():
    try:
        yield None
    finally:
        pass


def profile(
    *,
    print_to: Optional[Union[str, Path, IOBase]] = stdout,
    dump_to_path: Optional[Union[str, Path]] = None,
    encoding: str = 'utf8',
):
    """profile annotation to save profiling information of a function call

    Args:
        print_to (Optional[Union[str, Path, io.IOBase]], optional): file or path to print the results to. Defaults to stdout.
        dump_to_path (Optional[Union[str, Path]], optional): path to store the results at (in condensed form). Defaults to None.
        encoding (str, optional): file encoding to save the results to `print_to`. Ignored if `print_to` is not a `str`. Defaults to 'utf8'.

    Raises:
        TypeError: If the parameters' type don't match the signature

    Usage:
    ```
    @profile()
    def to_be_profiled(a, b, /, c, d, *, e, f):
        for _ in range(int(1e8)):
            pass

    to_be_profiled(1, 2, 3, d=4, e=5, f=6)
    ```
    """

    # pathlib.Path -> str
    if isinstance(print_to, Path):
        print_to = str(print_to)
    if isinstance(dump_to_path, Path):
        dump_to_path = str(dump_to_path)

    # build context manager for the textual output
    if isinstance(print_to, str):
        print_to_cm = open(print_to, mode='w', encoding=encoding)
    elif isinstance(print_to, IOBase):
        print_to_cm = print_to
    elif isinstance(print_to, type(None)):
        print_to_cm = _dummy_cm()
    else:
        raise TypeError(f'invalid value {print_to}')

    # type check the dump path
    if not isinstance(dump_to_path, (str, type(None))):
        raise TypeError(f'invalid value {dump_to_path}')

    def _profile(function):

        @wraps(function)
        def _wrapped(*args, **kwargs):
            with print_to_cm as dest:
                # actual profiling
                with cProfile.Profile() as pr:
                    try:
                        function(*args, **kwargs)
                    except Exception:
                        return

                # gather and sort stats
                stats = pstats.Stats(pr, stream=dest)
                stats.sort_stats(pstats.SortKey.TIME)

                # textual representation
                if dest is not None:
                    stats.print_stats()

                # condensed representation
                if dump_to_path is not None:
                    stats.dump_stats(dump_to_path)

        return _wrapped
    return _profile
