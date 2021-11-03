"""benchmark data for the meeting of november 19th

for each given imzML file:
    - convert it to zarr in a few different options (time once, no need for proper benchmark)
    - benchmark sum access
    - benchmark band access

save all data inside a shelve.Shelf (useful in case of late failure)
"""

import datetime
import itertools
import pathlib
import shelve
import sys
import timeit
import uuid
import warnings

from ...convert.imzml_to_zarr import converter
from ...benchmark.imzml import ImzMLBandBenchmark, ImzMLSumBenchmark
from ...benchmark.zarr_imzml import ZarrImzMLBandBenchmark, ZarrImzMLSumBenchmark

if len(sys.argv) == 1:
    sys.exit('expected at least one arguments')

_DB_DIR = pathlib.Path(__file__).resolve().parent / 'results'
if not _DB_DIR.exists():
    _DB_DIR.mkdir()

# conversion isn't monitored as the rechunking is not in its final form
_CONVERSION_NUMBER = 1
_CONVERSION_REPEAT = 1

_ACCESS_NUMBER = 100
_ACCESS_REPEAT = 5

_CONVERSION_KEY = 'conversion'
_ACCESS_IMZML_KEY = 'imzml_raw'
_ACCESS_ZARR_KEY = 'imzml_zarr'


def _make_db(file: str) -> shelve.Shelf:
    now = datetime.datetime.now()
    key = (file, now)
    shelf = shelve.open(str(_DB_DIR / str(hash(key))))

    shelf['key'] = key
    shelf[_ACCESS_IMZML_KEY] = {}
    shelf[_ACCESS_ZARR_KEY] = {}
    shelf[_CONVERSION_KEY] = {}

    return shelf


def _run(file: str) -> None:
    if file[-6:].lower() != '.imzml':
        raise ValueError(f'not an imzML file: {file}')
    zarr_path = f'{file[:-6]}_{uuid.uuid4()}.zarr'

    shelf = _make_db(file)

    tile_size = ((16, 16), (32, 32), (64, 64),  # small tiless
                 (256, 256), (512, 512),  # network-sized tiles
                 (1024, 1024), (2048, 2048))  # large tiles (?)

    # TODO support overlap in sum benchmark
    overlap_options = ((0, 0))  # no overlap for now
    # overlap_options += ((0, 1), (1, 0), (1, 1))

    # thin, auto or max NOTE (-1, -1, -1) is not supported by all compressors
    chunk_options = ((1, 1), True, (-1, -1, 1))

    order_options = ('C', 'F')

    compressor_options = ('default', None)

    options = itertools.product(
        chunk_options, order_options, compressor_options)
    tile_options = itertools.product(tile_size, overlap_options)

    # benchmark imzML raw - band access
    results = timeit.Timer(ImzMLBandBenchmark(file).task).repeat(
        _ACCESS_REPEAT, _ACCESS_NUMBER)
    tmp = shelf[_ACCESS_IMZML_KEY]
    tmp['band'] = results
    shelf[_ACCESS_IMZML_KEY] = tmp

    # benchmark imzML raw - tile access
    for tile, overlap in tile_options:
        benchmark = ImzMLSumBenchmark(file, tile)
        if hasattr(benchmark, 'broken'):
            continue
        results = timeit.Timer(benchmark.task).repeat(
            _ACCESS_REPEAT, _ACCESS_NUMBER)
        tmp = shelf[_ACCESS_IMZML_KEY]
        tmp[(tile, overlap)] = results
        shelf[_ACCESS_IMZML_KEY] = tmp

    for chunk, order, compressor in options:
        # conversion
        conversion = converter(file, zarr_path, chunks=chunk,
                               compressor=compressor, cache_metadata=True, order=order)
        results = timeit.Timer(conversion).repeat(
            _CONVERSION_REPEAT, _CONVERSION_NUMBER)

        # write data to log
        tmp = shelf[_CONVERSION_KEY]
        tmp[(chunk, order, compressor)] = results
        shelf[_CONVERSION_KEY] = tmp

        # benchmark converted file - band access
        results = timeit.Timer(ZarrImzMLBandBenchmark(zarr_path).task).repeat(
            _ACCESS_REPEAT, _ACCESS_NUMBER)
        tmp = shelf[_ACCESS_ZARR_KEY]
        tmp['band'] = results
        shelf[_ACCESS_ZARR_KEY] = tmp

        # benchmark converted file - tile access
        for tile, overlap in tile_options:
            benchmark = ZarrImzMLSumBenchmark(file, tile)
            if hasattr(benchmark, 'broken'):
                continue
            results = timeit.Timer(benchmark.task).repeat(
                _ACCESS_REPEAT, _ACCESS_NUMBER)
            tmp = shelf[_ACCESS_ZARR_KEY]
            tmp[(tile, overlap)] = results
            shelf[_ACCESS_ZARR_KEY] = tmp

    shelf.close()


# remove warnings emitted by pyimzml
warnings.filterwarnings('ignore')

for _file in sys.argv[1:]:
    _run(_file)
