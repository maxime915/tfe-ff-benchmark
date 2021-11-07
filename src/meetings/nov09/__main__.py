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
import shutil
import sys
import timeit
import uuid
import warnings

import zarr

from ...benchmark.imzml import ImzMLBandBenchmark, ImzMLSumBenchmark
from ...benchmark.zarr_imzml import (ZarrImzMLBandBenchmark, ZarrImzMLOverlapSumBenchmark,
                                     ZarrImzMLSumBenchmark)
from ...convert.imzml_to_zarr import converter

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
_ACCESS_ZARR_OVERLAP_KEY = 'imzml_zarr_overlap'


def _make_db(file: str) -> shelve.Shelf:
    now = datetime.datetime.now()
    key = (file, now)
    shelf = shelve.open(str(_DB_DIR / str(hash(key))))

    shelf['file'] = file
    shelf['date'] = now
    shelf['key'] = key
    shelf['parameters'] = {
        'conversion_number': _CONVERSION_NUMBER,
        'conversion_repeat': _CONVERSION_REPEAT,
        'access_number': _ACCESS_NUMBER,
        'access_repeat': _ACCESS_REPEAT
    }
    shelf[_ACCESS_IMZML_KEY] = {}
    shelf[_ACCESS_ZARR_KEY] = {}
    shelf[_CONVERSION_KEY] = {}
    shelf[_ACCESS_ZARR_OVERLAP_KEY] = {}

    return shelf


def _run(imzml_file: str) -> None:
    if imzml_file[-6:].lower() != '.imzml':
        raise ValueError(f'not an imzML file: {imzml_file}')
    zarr_path = f'{imzml_file[:-6]}_{uuid.uuid4()}.zarr'

    shelf = _make_db(imzml_file)

    tile_size = ((16, 16), (32, 32), (64, 64),  # small tiless
                 (256, 256), (512, 512),  # network-sized tiles
                 (1024, 1024), (2048, 2048))  # large tiles (?)

    # support overlap in sum benchmark
    overlap_options = ((0, 0), (0, 1), (1, 0), (1, 1))

    # thin, auto or max NOTE (-1, -1, -1) is not supported by all compressors
    chunk_options = ((1, 1), True, (-1, -1, 1))

    order_options = ('C', 'F')

    compressor_options = ('default', None)

    options = itertools.product(
        chunk_options, order_options, compressor_options)

    # benchmark imzML raw - band access
    results = timeit.Timer(ImzMLBandBenchmark(imzml_file).task).repeat(
        _ACCESS_REPEAT, _ACCESS_NUMBER)
    tmp = shelf[_ACCESS_IMZML_KEY]
    tmp['band'] = results
    shelf[_ACCESS_IMZML_KEY] = tmp

    # benchmark imzML raw - tile access
    for tile in tile_size:
        benchmark = ImzMLSumBenchmark(imzml_file, tile)
        if hasattr(benchmark, 'broken'):
            continue
        results = timeit.Timer(benchmark.task).repeat(
            _ACCESS_REPEAT, _ACCESS_NUMBER)
        tmp = shelf[_ACCESS_IMZML_KEY]
        tmp[tile] = results
        shelf[_ACCESS_IMZML_KEY] = tmp

    for chunk, order, compressor in options:
        # conversion
        conversion = converter(imzml_file, zarr_path, chunks=chunk,
                               compressor=compressor, cache_metadata=True, order=order)
        results = timeit.Timer(conversion).repeat(
            _CONVERSION_REPEAT, _CONVERSION_NUMBER)

        # write data to log
        tmp = shelf[_CONVERSION_KEY]
        tmp[(chunk, order, compressor, 'time')] = results
        tmp[(chunk, order, compressor, 'infos')] = zarr.open_array(
            zarr_path + '/intensities').info_items()
        shelf[_CONVERSION_KEY] = tmp

        # benchmark converted file - band access
        results = timeit.Timer(ZarrImzMLBandBenchmark(zarr_path).task).repeat(
            _ACCESS_REPEAT, _ACCESS_NUMBER)
        tmp = shelf[_ACCESS_ZARR_KEY]
        tmp[(chunk, order, compressor, 'band')] = results
        shelf[_ACCESS_ZARR_KEY] = tmp

        # benchmark converted file - tile access
        for tile in tile_size:
            benchmark = ZarrImzMLSumBenchmark(zarr_path, tile)
            if hasattr(benchmark, 'broken'):
                continue
            results = timeit.Timer(benchmark.task).repeat(
                _ACCESS_REPEAT, _ACCESS_NUMBER)
            tmp = shelf[_ACCESS_ZARR_KEY]
            tmp[(chunk, order, compressor, tile)] = results
            shelf[_ACCESS_ZARR_KEY] = tmp

            for overlap in overlap_options:
                benchmark = ZarrImzMLOverlapSumBenchmark(
                    zarr_path, tile, overlap)
                if hasattr(benchmark, 'broken'):
                    continue
                results = timeit.Timer(benchmark.task).repeat(
                    _ACCESS_REPEAT, _ACCESS_NUMBER)
                tmp = shelf[_ACCESS_ZARR_OVERLAP_KEY]
                tmp[(chunk, order, compressor, tile, overlap)] = results
                shelf[_ACCESS_ZARR_OVERLAP_KEY] = tmp

        shutil.rmtree(zarr_path)

    shelf.close()


# remove warnings emitted by pyimzml
warnings.filterwarnings('ignore')

for _file in sys.argv[1:]:
    _run(_file)
