"""start_benchmark: start the main benchmark (using conversion and access)

python -m src.start_benchmark file [file ...]

file is an HDF5 file to be converted to zarr, then multiple benchmark will be
run against both version

HDF5 profiles are encoded in X,Y,Z axis order using C order by default
Zarr only convert to X,Y,Z axis order but there is a Fortran order option
"""

import itertools
import shutil
import sys
import timeit
import uuid

import numpy

from ...benchmark.tile3d_zarr import run_benchmark as zarr_benchmark
from ...benchmark.tile3d_hdf5 import run_benchmark as hdf5_benchmark
from ...convert.companion_to_zarr import converter

_CONVERSION_NUMBER = 2
_CONVERSION_REPEAT = 3

_ACCESS_NUMBER = 200
_ACCESS_REPEAT = 5


def _run(hdf5_path: str):
    return
    if hdf5_path[-5:] != '.hdf5':
        raise ValueError(f'not a HDF5 file :{hdf5_path}')
    zarr_path = f'{hdf5_path[:-5]}_{uuid.uuid4()}.zarr'

    # (128, 128, 1) to (1024, 1024, 1)
    # (128, 128, all) to (512, 512, all)
    tile_options = ([[2**i, 2**i, 1] for i in range(7, 11)]
                    + [[2**i, 2**i, -1] for i in range(7, 10)])

    # NOTE using a chunk value too low (32, 32, 1) can cause some memory issues
    # (128, 128, 1) to (1024, 1024, 1)
    # (128, 128, all) to (512, 512, all)
    chunk_options = ([[2**i, 2**i, 1] for i in range(7, 11)]
                     + [[2**i, 2**i, -1] for i in range(7, 10)])

    order_options = ['C', 'F']

    compressor_options = ['default', None]

    options = itertools.product(
        chunk_options, order_options, compressor_options)

    for tile in tile_options:
        # benchmark HDF5 access
        hdf5_benchmark([hdf5_path], _ACCESS_NUMBER, _ACCESS_REPEAT, tile=tile)
        sys.stdout.flush()

    for chunk, order, compressor in options:
        print(
            f'converting {hdf5_path} to {zarr_path} ({chunk=}, {order=}, {compressor=})')

        # make zarr conversion
        conversion = converter(hdf5_path, zarr_path, chunks=chunk,
                               compressor=compressor, cache_metadata=True, order=order)
        results = numpy.array(timeit.Timer(conversion).
                              repeat(_CONVERSION_REPEAT, number=_CONVERSION_NUMBER)) / _CONVERSION_NUMBER
        print(f'\t[{", ".join([f"{v:5.3e}" for v in results])}]')  # whole data
        print(f'\t{results.min() = :5.3e} s | {results.max() = :5.3e} s')
        print(
            f'\tmean = {results.mean():5.3e} s ± std = {results.std(ddof=1):5.3e} s')
        print('')  # add newline
        sys.stdout.flush()

        for tile in tile_options:
            # benchmark zarr access
            zarr_benchmark([zarr_path], _ACCESS_NUMBER,
                           _ACCESS_REPEAT, tile=tile)
            sys.stdout.flush()
        
    shutil.rmtree(zarr_path)


if __name__ == "__main__":
    for file in sys.argv[1:]:
        _run(file)
else:
    raise ImportError("module should not be imported")
