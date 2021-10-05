"""start_benchmark: start the main benchmark (using conversion and access)

python -m src.start_benchmark file [file ...]

file is an HDF5 file to be converted to zarr, then multiple benchmark will be
run against both version
"""

import itertools
import sys
import timeit

import numpy

from .benchmark.tile3d_zarr import run_benchmark as zarr_benchmark
from .benchmark.tile3d_hdf5 import run_benchmark as hdf5_benchmark
from .convert.companion_to_zarr import converter


def _run(hdf5_path: str):
    if hdf5_path[-5:] != '.hdf5':
        raise ValueError(f'not a HDF5 file :{hdf5_path}')
    zarr_path = hdf5_path[:-5] + '.zarr'

    tile_options = ([[2**i, 2**i, 1] for i in range(0, 12)]
                    + [[2**i, 2**i, -1] for i in range(5, 12)])
    chunk_options = ([True, None]
                     + [[2**i, 2**i, 1] for i in range(5, 12)]
                     + [[2**i, 2**i, -1] for i in range(5, 10)])
    order_options = ['C', 'F']
    compressor_options = ['default', None]

    options = itertools.product(
        chunk_options, order_options, compressor_options)

    for tile in tile_options:
        for chunk, order, compressor in options:
            print(f'converting {hdf5_path} to {zarr_path} ({chunk=}, {order=}, {compressor=})')

            # make zarr conversion
            conversion = converter(hdf5_path, zarr_path, chunks=chunk,
                                   compressor=compressor, cache_metadata=True, order=order)
            results = numpy.array(timeit.Timer(conversion).repeat(3, number=10)) / 10
            print(f'\t[{", ".join([f"{v:5.3e}" for v in results])}]') # whole data
            print(f'\t{results.min() = :5.3e} s | {results.max() = :5.3e} s')
            print(f'\tmean = {results.mean():5.3e} s ± std = {results.std(ddof=1):5.3e} s')
            print('') # add newline

            # benchmark zarr access
            zarr_benchmark([zarr_path], 1000, 5, tile=tile)

        # benchmark HDF5 access
        hdf5_benchmark([hdf5_path], 1000, 5, tile=tile)


if __name__ == "__main__":
    for file in sys.argv[1:]:
        _run(file)
else:
    raise ImportError("module should not be imported")
