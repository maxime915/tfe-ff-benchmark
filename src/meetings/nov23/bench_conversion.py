"""bench_conversion: check the different conversion method
"""

import itertools
import multiprocessing
import os
import shutil
import sys
import time
import typing
import uuid
import warnings
from timeit import Timer, default_timer

import zarr

from ...benchmark.imzml import imzml_path_to_info
from ...convert.imzml_to_zarr import converter
from .db import get_db_dir, get_db_for_file

if len(sys.argv) == 1:
    sys.exit('expected at least one arguments')

# conversion isn't averaged (takes too much time for this benchmark)
_CONVERSION_NUMBER = 1
_CONVERSION_REPEAT = 2


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

            results = Timer(bench).repeat(repeat, number)

            cpu_time_end = time.process_time()

            pipe_left.send(results)
            pipe_left.send(cpu_time_end - cpu_time_start)
        except Exception as e:
            pipe_left.send(str(e))

    # new process to avoid everything crashing if OOMKiller wakes up
    process = multiprocessing.Process(target=body)

    # record wall time in parent process in case of error: better than nothing
    wall_time_start = default_timer()

    # start benchmark and wait for results
    process.start()
    process.join()

    wall_time = default_timer() - wall_time_start

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
    print(f'\tdone! (WTime: {wall_time: 5.2f}, CPUTime: {proc_time: 5.2f}')
    sys.stdout.flush()
    return results


def _run(imzml_file: str) -> None:
    cpu_time_start = time.process_time()
    wall_time_start = default_timer()

    if imzml_file[-6:].lower() != '.imzml':
        raise ValueError(f'not an imzML file: {imzml_file}')
    zarr_path = f'{imzml_file[:-6]}_{uuid.uuid4()}.zarr'

    store = get_db_for_file(imzml_file, get_db_dir(
        __file__, 'conversion_results'))
    store.save_val_at({
        'conversion number': _CONVERSION_NUMBER,
        'conversion repeat': _CONVERSION_REPEAT,
    }, 'benchmark infos')

    # store more file infos
    imzml_info = imzml_path_to_info(imzml_file)
    store.save_val_at(imzml_info, 'imzml info')

    order_choice = ('C', 'F')
    compressor_choice = ('default', None)
    chunk_choice = ((1, 1, -1), True, (-1, -1, 1))
    rechunker_choice = (True, False)
    max_mem_choice = (2**30, 2**31, 2**32)  # 1GiB, 2GiB, 4GiB

    # chunks option are not the same on processed files (ragged arrays)
    if not imzml_info.continuous_mode:
        # thin, guessed with average band size, full
        chunk_choice = (
            (1, 1),
            True,
            (-1, -1),
        )

    options = itertools.product(chunk_choice, compressor_choice, order_choice,
                                rechunker_choice, max_mem_choice)

    for (chunk, compressor, order, rechunk, max_mem) in options:
        # base key for this converted file
        base_key = (chunk, compressor, order, rechunk, max_mem)

        # conversion
        conversion = converter(imzml_file, zarr_path, chunks=chunk,
                               compressor=compressor, cache_metadata=True, order=order,
                               max_mem=max_mem, use_rechunker=rechunk)
        results = _bench(conversion, f'conversion {base_key=}', _CONVERSION_REPEAT, _CONVERSION_NUMBER)

        if results and results[0] >= 0:
            # success !
            
            # store wall time
            store.save_val_at(results, *base_key, 'conversion time')
            
            # infos
            infos_int = zarr.open_array(zarr_path + '/intensities').info_items()
            store.save_val_at(infos_int, *base_key, 'infos', 'intensities')

            infos_mzs = zarr.open_array(zarr_path + '/mzs').info_items()
            store.save_val_at(infos_mzs, *base_key, 'infos', 'mzs')
        else:
            # failure
            
            # keep the [-1] as an indication
            store.save_val_at(results, *base_key, 'conversion time')
            store.save_val_at('no info: failed', *base_key,'infos', 'intensities')
            store.save_val_at('no info: failed', *base_key, 'infos', 'mzs')

        shutil.rmtree(zarr_path)

    cpu_time_end = time.process_time()
    wall_time_end = default_timer()

    print(f'Conversion on {imzml_file}')
    print(f'Total wall time: {wall_time_end-wall_time_start}')
    print(f'Total CPU time: {cpu_time_end-cpu_time_start}')


# remove warnings emitted by pyimzml about accession typos (they)
warnings.filterwarnings('ignore', message=r'.*Accession IMS.*')

print(f'{os.getpid()=}')
for _file in sys.argv[1:]:
    _run(_file)
