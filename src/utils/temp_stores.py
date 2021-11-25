"temp_stores: wrapper around zarr.TempStore to provide automatic removal"

import contextlib

import zarr


@contextlib.contextmanager
def temp_stores(*args, count: int = 1, **kw):
    """temp stores: context manager to generate multiple zarr.TempStore and close
    them as soon as they are not needed anymore

    Args:
        count (int, optional): Number of store to create. Defaults to 1.
        *args and **kw are passed to zarr.TempStore(...)

    Yields:
        Tuple[Zarr.TempStore]: a collection of stores
    """
    with curried_temp_stores(*args, **kw)(count) as stores:
        yield stores


def curried_temp_stores(*args, **kw):
    """curried_temp_stores: returns a context manager that generates multiple
    zarr.TempStore and closes them as soon as they are not needed anymore

    Args are passed to zarr.TempStore(...) each time a new store is created

    Returns: a context manager that takes an optional count: int argument
        (default: 1) for the number of TempStore to open.

    Usage:
    ```
    get_temp_stores = curried_temp_stores('suffix', normalize_keys=False)
    with get_temp_stores(2) as (temp1, temp2):
        do_something(temp1)
        do_something(temp2)
    # or
    with get_temp_stores(count=n) as store_tpl: pass
    # or
    with get_temp_stores() as store: pass
    ```
    """

    @contextlib.contextmanager
    def get_temp_stores(count: int = 1):
        stores = ()
        try:
            assert count > 0
            stores = tuple(zarr.TempStore(*args, **kw) for _ in range(count))
            yield stores
        finally:
            for store in stores:
                # safe to call, even if the store was removed from disk
                store.rmdir()

    return get_temp_stores
