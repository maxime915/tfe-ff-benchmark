"""benchmark data for the meeting of november 19th

for each given imzML file:
    - convert it to zarr in a few different options (time once, no need for proper benchmark)
    - benchmark sum access
    - benchmark band access

save all data inside a shelve.Shelf (useful in case of late failure)
"""

import contextlib
import logging
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
from ...convert.imzml_to_zarr.one_phase import convert
from .db import get_db_for_file

if len(sys.argv) == 1:
    sys.exit("expected at least one arguments")

# conversion isn't measured this time

# imzML has a really high latency, precision isn't has important
_IMZML_NUMBER = 10
_IMZML_REPEAT = 2

_ZARR_NUMBER = 100
_ZARR_REPEAT = 3

_IMZML_KEY = "imzml_raw"
_ZARR_KEY = "imzml_zarr"

MAX_MEM = 4 * 2**30  # 4GiB

#_IMZML_NUMBER = 1
#_IMZML_REPEAT = 2
#_ZARR_NUMBER = 1
#_ZARR_REPEAT = 2


def _bench(
    bench: typing.Callable[[], None], name: str, repeat: int, number: int
) -> typing.List[float]:
    """_bench: run a benchmark of *bench* in a new process

    - name: str to log at the beginning
    - repeat, number: see timeit.repeat
    - bench: function to benchmark

    returns: a list of float if OK, [-1] otherwise"""

    logging.info(f"doing {name}...")
    sys.stdout.flush()

    # send results and CPU time
    pipe_left, pipe_right = multiprocessing.Pipe()

    # function to run the benchmark into
    def body():
        try:
            # inside of a new process -> no impact to the main one
            warnings.filterwarnings("error")
            # avoid these two
            warnings.filterwarnings("ignore", message=r"Accession I?MS")  # pyImzML bug
            warnings.filterwarnings("ignore", message=r"tiles=.* too large")  # anticipated
            warnings.filterwarnings("ignore", message=r"chunks=.* too large")  # anticipated

            cpu_time_start = time.process_time()

            results = timeit.Timer(bench).repeat(repeat, number)

            cpu_time_end = time.process_time()

            pipe_left.send(results)
            pipe_left.send(cpu_time_end - cpu_time_start)
        except Exception as e:
            logging.exception(f'EXCEPTION_ID={id(e)}')
            pipe_left.send(id(e))

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
        logging.info(f"\tfailed after {wall_time: 5.2f}, {process.exitcode=}")
        sys.stdout.flush()
        return [-1]

    # if excpetion aborted benchmark, receive a int as id
    exception = pipe_right.recv()
    if isinstance(exception, int):
        logging.info(f"\tfailed after {wall_time: 5.2f}, see EXCEPTION_ID={exception}")
        sys.stdout.flush()
        return [-1]

    # everything went fine, receive results and CPU time
    results = exception
    proc_time = pipe_right.recv()
    logging.info(f"\tdone! (WTime: {wall_time: 5.2f}, CPUTime: {proc_time: 5.2f})")
    sys.stdout.flush()
    return results


def _run(imzml_file: str) -> None:
    wall_time_start = timeit.default_timer()

    if imzml_file[-6:].lower() != ".imzml":
        raise ValueError(f"not an imzML file: {imzml_file}")
    zarr_path = f"{imzml_file[:-6]}_{uuid.uuid4()}.zarr"

    store = get_db_for_file(imzml_file, db_dir=__file__)
    store.save_val_at(
        {
            "imzml number": _IMZML_NUMBER,
            "imzml repeat": _IMZML_REPEAT,
            "zarr number": _ZARR_NUMBER,
            "zarr repeat": _ZARR_REPEAT,
        },
        "benchmark infos",
    )

    logging.info(f"results stored into {store.path}")

    # store more file infos
    imzml_info = imzml_path_to_info(imzml_file)
    store.save_val_at(imzml_info, "imzml info")

    # parameter to vary during the benchmark
    window_choice = (
        (16, 16),
        (32, 32),
        (64, 64),  # small windows
        (128, 128),
        (256, 256),
        (512, 512),  # network-sized windows
        (1024, 1024),
        (2048, 2048),
    )  # large windows (?)
    overlap_choice = ((0, 0), (0, 1), (1, 0), (1, 1))
    chunk_choice = [
        "dask-auto",
        "dask-auto-small",
        "zarr-auto",
        "dask-spectral",
        "dask-spatial",
    ]

    # save all choice option into the db for further analysis
    store.save_val_at(
        {
            "tile_choice": window_choice,
            "overlap_choice": overlap_choice,
            "chunk_choice": chunk_choice,
        },
        "benchmark parameters",
    )

#    # benchmark imzML raw - band access
#    results = _bench(
#        ImzMLBandBenchmark(imzml_file).task, "imzml: band", _IMZML_REPEAT, _IMZML_NUMBER
#    )
#    store.save_val_at(results, _IMZML_KEY, "band")

#    # benchmark imzML raw - tile access (TIC & search)
#    for tile in window_choice:
#        benchmark = ImzMLSumBenchmark(imzml_file, tile)
#        if hasattr(benchmark, "broken"):
#            continue

#        results = _bench(
#            benchmark.task, f"imzml TIC {tile}", _IMZML_REPEAT, _IMZML_NUMBER
#        )
#        store.save_val_at(results, _IMZML_KEY, "tic", tile)

#        benchmark = ImzMLSearchBenchmark(imzml_file, tile, imzml_info)
#        if hasattr(benchmark, "broken"):
#            continue

#        results = _bench(
#            benchmark.task, f"imzml search {tile}", _IMZML_REPEAT, _IMZML_NUMBER
#        )
#        store.save_val_at(results, _IMZML_KEY, "search", tile)

    imzml_path = pathlib.Path(imzml_file)

    for chunk in chunk_choice:
        with contextlib.ExitStack() as stack:
            # base key for this converted file
            base_key = (_ZARR_KEY, chunk)

            stack.callback(shutil.rmtree, zarr_path, ignore_errors=True)

            status = convert(
                imzml_path,
                imzml_path.with_suffix(".ibd"),
                pathlib.Path(zarr_path),
                compressor="default",
                chunks=chunk,
                order="C",
                max_size=MAX_MEM,
            )

            # handle conversion failure
            if not status:
                store.save_val_at("no info: failed", *base_key, "infos", "intensities")
                store.save_val_at("no info: failed", *base_key, "infos", "mzs")
                continue

            # infos
            infos_int = zarr.open_array(zarr_path + "/0").info_items()
            store.save_val_at(infos_int, *base_key, "infos", "intensities")

            infos_mzs = zarr.open_array(zarr_path + "/labels/mzs/0").info_items()
            store.save_val_at(infos_mzs, *base_key, "infos", "mzs")

            logging.info(f"{chunk = }")

            # benchmark converted file - band access
            results = _bench(
                ZarrImzMLBandBenchmark(zarr_path).task,
                "zarr band",
                _ZARR_REPEAT,
                _ZARR_NUMBER,
            )
            store.save_val_at(results, *base_key, "band")

            # benchmark converted file - tile access
            for tile in window_choice:
                benchmark = ZarrImzMLSumBenchmark(zarr_path, tile)
                if hasattr(benchmark, "broken"):
                    continue
                results = _bench(
                    benchmark.task, f"zarr tic {tile}", _ZARR_REPEAT, _ZARR_NUMBER
                )
                store.save_val_at(results, *base_key, "tic", tile)

                benchmark = ZarrImzMLSearchBenchmark(zarr_path, tile, imzml_info)
                if hasattr(benchmark, "broken"):
                    continue
                results = _bench(
                    benchmark.task, f"zarr search {tile}", _ZARR_REPEAT, _ZARR_NUMBER
                )
                store.save_val_at(results, *base_key, "search", tile)

                # benchmark converted file - tile access with overlap
                for overlap in overlap_choice:
                    benchmark = ZarrImzMLOverlapSumBenchmark(zarr_path, tile, overlap)
                    if hasattr(benchmark, "broken"):
                        continue
                    results = _bench(
                        benchmark.task,
                        f"zarr tic {tile} {overlap}",
                        _ZARR_REPEAT,
                        _ZARR_NUMBER,
                    )
                    store.save_val_at(results, *base_key, "tic-overlap", tile, overlap)

    wall_time_end = timeit.default_timer()

    logging.info(f"ImzML / Zarr benchmark on {imzml_file}")
    logging.info(f"Total wall time: {wall_time_end-wall_time_start}")

# logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=0)

# remove warnings emitted by pyimzml about accession typos
warnings.filterwarnings("ignore", message=r"Accession I?MS")
warnings.filterwarnings("ignore", message=r"tiles=.* too large")

logging.info(f"starting benchmark at PID: {os.getpid()}")
for _file in sys.argv[1:]:
    _run(_file)
