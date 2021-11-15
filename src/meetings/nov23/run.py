"""benchmark data for the meeting of november 19th

for each given imzML file:
    - convert it to zarr in a few different options (time once, no need for proper benchmark)
    - benchmark sum access
    - benchmark band access

save all data inside a shelve.Shelf (useful in case of late failure)
"""

import itertools
import shutil
import sys
import time
import timeit
import uuid
import warnings

import zarr

from ...benchmark.imzml import (ImzMLBandBenchmark, ImzMLSearchBenchmark,
                                ImzMLSumBenchmark, imzml_path_to_info)
from ...benchmark.zarr_imzml import (ZarrImzMLBandBenchmark,
                                     ZarrImzMLOverlapSumBenchmark,
                                     ZarrImzMLSearchBenchmark,
                                     ZarrImzMLSumBenchmark)
from ...convert.imzml_to_zarr import converter
from .db import get_db_for_file

if len(sys.argv) == 1:
    sys.exit('expected at least one arguments')


# conversion isn't averaged (previous experiments showed almost no variation)
_CONVERSION_NUMBER = 1
_CONVERSION_REPEAT = 2

# imzML has a really high latency, precision isn't has important
_IMZML_NUMBER = 20
_IMZML_REPEAT = 3

_ZARR_NUMBER = 100
_ZARR_REPEAT = 5

_IMZML_KEY = 'imzml_raw'
_ZARR_KEY = 'imzml_zarr'


def _run(imzml_file: str) -> None:
    cpu_time_start = time.process_time()
    wall_time_start = timeit.default_timer()

    if imzml_file[-6:].lower() != '.imzml':
        raise ValueError(f'not an imzML file: {imzml_file}')
    zarr_path = f'{imzml_file[:-6]}_{uuid.uuid4()}.zarr'

    store = get_db_for_file(imzml_file, db_dir=__file__)
    store.save_val_at({
        'conversion number': _CONVERSION_NUMBER,
        'conversion repeat': _CONVERSION_REPEAT,
        'imzml number': _IMZML_NUMBER,
        'imzml repeat': _IMZML_REPEAT,
        'zarr number': _ZARR_NUMBER,
        'zarr repeat': _ZARR_REPEAT,
    }, 'benchmark infos')

    # store more file infos
    imzml_info = imzml_path_to_info(imzml_file)
    store.save_val_at(imzml_info, 'imzml info')

    # parameter to vary during the benchmark
    tile_choice = ((16, 16), (32, 32), (64, 64),  # small tiless
                   (256, 256), (512, 512),  # network-sized tiles
                   (1024, 1024), (2048, 2048))  # large tiles (?)
    overlap_choice = ((0, 0), (0, 1), (1, 0), (1, 1))
    order_choice = ('C', 'F')
    compressor_choice = ('default', None)
    chunk_choice = ((1, 1, -1), True, (-1, -1, 1))

    # chunks option are not the same on processed files (ragged arrays)
    if not imzml_info.continuous_mode:
        # thin, guessed with average band size, full
        chunk_choice = (
            (1, 1),
            True,
            (-1, -1),
        )

    options = itertools.product(chunk_choice, order_choice, compressor_choice)

    # benchmark imzML raw - band access
    results = timeit.Timer(ImzMLBandBenchmark(imzml_file).task).repeat(
        _IMZML_REPEAT, _IMZML_NUMBER)
    store.save_val_at(results, _IMZML_KEY, 'band')

    # benchmark imzML raw - tile access (TIC & search)
    for tile in tile_choice:
        benchmark = ImzMLSumBenchmark(imzml_file, tile)
        if hasattr(benchmark, 'broken'):
            continue

        results = timeit.Timer(benchmark.task).repeat(
            _IMZML_REPEAT, _IMZML_NUMBER)
        store.save_val_at(results, _IMZML_KEY, 'tic', tile)

        benchmark = ImzMLSearchBenchmark(imzml_file, tile, imzml_info)
        if hasattr(benchmark, 'broken'):
            continue

        results = timeit.Timer(benchmark.task).repeat(
            _IMZML_REPEAT, _IMZML_NUMBER)
        store.save_val_at(results, _IMZML_KEY, 'search', tile)

    for chunk, order, compressor in options:
        # base key for this converted file
        base_key = (_ZARR_KEY, chunk, compressor, order)

        # conversion
        conversion = converter(imzml_file, zarr_path, chunks=chunk,
                               compressor=compressor, cache_metadata=True, order=order)
        results = timeit.Timer(conversion).repeat(
            _CONVERSION_REPEAT, _CONVERSION_NUMBER)
        store.save_val_at(results, *base_key, 'conversion time')

        # infos
        infos_int = zarr.open_array(zarr_path + '/intensities').info_items()
        store.save_val_at(infos_int, *base_key, 'infos', 'intensities')

        infos_mzs = zarr.open_array(zarr_path + '/mzs').info_items()
        store.save_val_at(infos_mzs, *base_key, 'infos', 'mzs')

        print(f'{chunk = }, {order = }, {compressor = }')

        # benchmark converted file - band access
        results = timeit.Timer(ZarrImzMLBandBenchmark(zarr_path).task).repeat(
            _ZARR_REPEAT, _ZARR_NUMBER)
        store.save_val_at(results, *base_key, 'band')

        # benchmark converted file - tile access
        for tile in tile_choice:
            benchmark = ZarrImzMLSumBenchmark(zarr_path, tile)
            if hasattr(benchmark, 'broken'):
                continue
            results = timeit.Timer(benchmark.task).repeat(
                _ZARR_REPEAT, _ZARR_NUMBER)
            store.save_val_at(results, *base_key, 'tic', tile)

            benchmark = ZarrImzMLSearchBenchmark(zarr_path, tile, imzml_info)
            if hasattr(benchmark, 'broken'):
                continue
            results = timeit.Timer(benchmark.task).repeat(
                _ZARR_REPEAT, _ZARR_NUMBER)
            store.save_val_at(results, *base_key, 'search', tile)

            # benchmark converted file - tile access with overlap
            for overlap in overlap_choice:
                benchmark = ZarrImzMLOverlapSumBenchmark(
                    zarr_path, tile, overlap)
                if hasattr(benchmark, 'broken'):
                    continue
                results = timeit.Timer(benchmark.task).repeat(
                    _ZARR_REPEAT, _ZARR_NUMBER)
                store.save_val_at(results, *base_key,
                                  'tic-overlap', tile, overlap)

        shutil.rmtree(zarr_path)

    cpu_time_end = time.process_time()
    wall_time_end = timeit.default_timer()

    print(f'ImzML / Zarr benchmark on {imzml_file}')
    print(f'Wall time: {wall_time_end-wall_time_start}')
    print(f'CPU time: {cpu_time_end-cpu_time_start}')


# remove warnings emitted by pyimzml about accession typos (they)
warnings.filterwarnings('ignore', message=r'.*Accession IMS.*')

for _file in sys.argv[1:]:
    _run(_file)
