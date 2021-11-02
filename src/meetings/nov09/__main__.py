"""benchmark data for the meeting of november 19th

for each given imzML file:
    - convert it to zarr in a few different options (time once, no need for proper benchmark)
    - benchmark sum access
    - benchmark band access

save all data inside a shelve.Shelf (useful in case of late failure)
"""

import datetime
import pathlib
import shelve
import sys
import tempfile
import timeit

from ...convert.imzml_to_zarr import converter

if len(sys.argv) == 1:
    sys.exit('expected at least one arguments')

_DB_DIR = pathlib.Path(__file__).resolve().parent / 'results'
if not _DB_DIR.exists():
    _DB_DIR.mkdir()


def _make_db(file: str) -> shelve.Shelf:
    now = datetime.datetime.now()
    key = (file, now)
    shelf = shelve.open(str(_DB_DIR / str(hash(key))))

    shelf['key'] = key
    return shelf


# TODO define options


def _run(file: str) -> None:
    shelf = _make_db(file)
    tmp_dir = tempfile.TemporaryDirectory()

    # conversion
    values = timeit.repeat(
        converter(file, str(pathlib.Path(tmp_dir.name) / 'c.zarr'), **{

        }),
        repeat=1, number=1,
    )

    shelf[('conversion')] = values

    # benchmark
    # TODO

    shelf.close()


for file in sys.argv[1:]:
    _run(file)
