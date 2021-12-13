"""benchmark data for the meeting of november 19th

for each given imzML file:
    - convert it to zarr in a few different options (time once, no need for proper benchmark)
    - benchmark sum access
    - benchmark band access

save all data inside a shelve.Shelf (useful in case of late failure)
"""

import contextlib
import itertools
import multiprocessing
import os
import pathlib
import shutil
import sys
import time
import timeit
import typing
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

MAX_MEM = 4 * 2**30  # 4GiB


# NOTE improvement idea: store repeat and number directly to avoid extra
#   boilerplate everywhere and store these numbers as well to make the tool more versatile
def _bench(bench: typing.Callable[[], None], name: str, repeat: int,
           number: int) -> typing.List[float]:
    """_bench: run a benchmark of *bench* in a new process

    - name: str to print at the beginning
    - repeat, number: see timeit.repeat
    - bench: function to benchmark

    returns: a list of float if OK, [-1] otherwise"""

    print(f'doing {name}...')
    sys.stdout.flush()

    # send results and CPU time
    pipe_left, pipe_right = multiprocessing.Pipe()

    # function to run the benchmark into
    def body():
        try:
            cpu_time_start = time.process_time()

            results = timeit.Timer(bench).repeat(repeat, number)

            cpu_time_end = time.process_time()

            pipe_left.send(results)
            pipe_left.send(cpu_time_end - cpu_time_start)
        except Exception as e:
            pipe_left.send(str(e))

    # new process to avoid everything crashing if OOMKiller wakes up
    process = multiprocessing.Process(target=body)

    # record wall time in parent process in case of error: better than nothing
    wall_time_start = timeit.default_timer()

    # start benchmark and wait for results
    process.start()
    process.join()

    wall_time = timeit.default_timer() - wall_time_start

    # if processed was killed, receive nothing.
    if process.exitcode < 0:
        print(f'\tfailed after {wall_time: 5.2f}, {process.exitcode=}')
        sys.stdout.flush()
        return [-1]

    # if excpetion aborted benchmark, receive a str with the message
    exception = pipe_right.recv()
    if isinstance(exception, str):
        print(f'\tfailed after {wall_time: 5.2f}, with {exception=}')
        sys.stdout.flush()
        return [-1]

    # everything went fine, receive results and CPU time
    results = exception
    proc_time = pipe_right.recv()
    print(f'\tdone! (WTime: {wall_time: 5.2f}, CPUTime: {proc_time: 5.2f})')
    sys.stdout.flush()
    return results


def _run(imzml_file: str) -> None:
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
    
    print(f'results stored into {store.path}')

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
    chunk_choice = ((-1, 1, 1, 1), True, (1, 1, -1, -1))

    # save all choice option into the db for further analysis
    store.save_val_at({
        'tile_choice': tile_choice,
        'overlap_choice': overlap_choice,
        'order_choice': order_choice,
        'compressor_choice': compressor_choice,
        'chunk_choice': chunk_choice
    }, 'benchmark parameters')

    options = itertools.product(chunk_choice, order_choice, compressor_choice)

    # benchmark imzML raw - band access
    results = _bench(ImzMLBandBenchmark(imzml_file).task,
                     'imzml: band', _IMZML_REPEAT, _IMZML_NUMBER)
    store.save_val_at(results, _IMZML_KEY, 'band')

    # benchmark imzML raw - tile access (TIC & search)
    for tile in tile_choice:
        benchmark = ImzMLSumBenchmark(imzml_file, tile)
        if hasattr(benchmark, 'broken'):
            continue

        results = _bench(
            benchmark.task, f'imzml TIC {tile}', _IMZML_REPEAT, _IMZML_NUMBER)
        store.save_val_at(results, _IMZML_KEY, 'tic', tile)

        benchmark = ImzMLSearchBenchmark(imzml_file, tile, imzml_info)
        if hasattr(benchmark, 'broken'):
            continue

        results = _bench(
            benchmark.task, f'imzml search {tile}', _IMZML_REPEAT, _IMZML_NUMBER)
        store.save_val_at(results, _IMZML_KEY, 'search', tile)

    for chunk, order, compressor in options:
        with contextlib.ExitStack() as stack:
            # base key for this converted file
            base_key = (_ZARR_KEY, chunk, compressor, order)

            # conversion
            # use rechunker for continuous mode and other method for processed mode
            conversion = converter(imzml_file, zarr_path, chunks=chunk,
                                   compressor=compressor, cache_metadata=True, order=order,
                                   use_rechunker=False,
                                   max_mem=MAX_MEM)

            stack.callback(shutil.rmtree, zarr_path, ignore_errors=True)
            
            results = _bench(
                conversion, f'conversion {chunk} {order}, {compressor}', _CONVERSION_REPEAT, _CONVERSION_NUMBER)
            store.save_val_at(results, *base_key, 'conversion time')

            # handle conversion failure
            if results == [-1]:
                store.save_val_at('no info: failed', *base_key,
                                  'infos', 'intensities')
                store.save_val_at('no info: failed', *base_key, 'infos', 'mzs')
                continue

            # infos
            infos_int = zarr.open_array(zarr_path + '/0').info_items()
            store.save_val_at(infos_int, *base_key, 'infos', 'intensities')

            infos_mzs = zarr.open_array(
                zarr_path + '/labels/mzs/0').info_items()
            store.save_val_at(infos_mzs, *base_key, 'infos', 'mzs')

            print(f'{chunk = }, {order = }, {compressor = }')

            # benchmark converted file - band access
            results = _bench(ZarrImzMLBandBenchmark(zarr_path).task,
                             'zarr band', _ZARR_REPEAT, _ZARR_NUMBER)
            store.save_val_at(results, *base_key, 'band')

            # benchmark converted file - tile access
            for tile in tile_choice:
                benchmark = ZarrImzMLSumBenchmark(zarr_path, tile)
                if hasattr(benchmark, 'broken'):
                    continue
                results = _bench(
                    benchmark.task, f'zarr tic {tile}', _ZARR_REPEAT, _ZARR_NUMBER)
                store.save_val_at(results, *base_key, 'tic', tile)

                benchmark = ZarrImzMLSearchBenchmark(
                    zarr_path, tile, imzml_info)
                if hasattr(benchmark, 'broken'):
                    continue
                results = _bench(
                    benchmark.task, f'zarr search {tile}', _ZARR_REPEAT, _ZARR_NUMBER)
                store.save_val_at(results, *base_key, 'search', tile)

                # benchmark converted file - tile access with overlap
                for overlap in overlap_choice:
                    benchmark = ZarrImzMLOverlapSumBenchmark(
                        zarr_path, tile, overlap)
                    if hasattr(benchmark, 'broken'):
                        continue
                    results = _bench(
                        benchmark.task, f'zarr tic {tile} {overlap}', _ZARR_REPEAT, _ZARR_NUMBER)
                    store.save_val_at(results, *base_key,
                                      'tic-overlap', tile, overlap)

    wall_time_end = timeit.default_timer()

    print(f'ImzML / Zarr benchmark on {imzml_file}')
    print(f'Total wall time: {wall_time_end-wall_time_start}')


# remove warnings emitted by pyimzml about accession typos
warnings.filterwarnings('ignore', message=r'.*Accession IMS.*')
warnings.filterwarnings('ignore', message=r'.*Accession MS.*')

print(f'starting benchmark at PID: {os.getpid()}')
for _file in sys.argv[1:]:
    _run(_file)
