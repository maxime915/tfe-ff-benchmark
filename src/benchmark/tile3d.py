"""tile3d_abc: benchmark files by loading a 3D tile

base work for 3D with overlap window benchmark

this is mostly the product of some reflection and probably shouldn't be used
"""

import abc
# import itertools
import pathlib
import random
import warnings
from typing import Iterable

import h5py
import zarr

from .benchmark_abc import BenchmarkABC, Verbosity


warnings.warn('tile3D should not be run or imported')


class Tile3DBenchmarkABC(BenchmarkABC):
    "access a multidimensional array with a random Nd slice"

    def __init__(self, path: str, tile: Iterable[int], overlap: Iterable[int]) -> None:
        """ - tile: tuple/list/etc. of same shape as array.shape, -1 for fractional size
        - overlap: tuple/list/etc. of same shape as array.shape, 1 for overlap between
        two chunks on this axis, 0 to be inside one chunk.

        tile[i] is suppose to be less than or equal to chunk[i] if chunk corresponds
        to the loaded array referenced by path.
        """
        self.path = path
        self.tile = tile
        self.overlap = overlap

    @abc.abstractmethod
    def get_array(self):
        "load the resource as an object that cn be indexed and has a shape"

    def task(self) -> None:
        array = self.get_array()

        # fraction tile: set -2 to size/2, etc
        tile = [t if t > 0 else s // -t for (s, t)
                in zip(array.shape, self.tile)]
        # array.shape[i] >= tile[i] > 0
        # we can suppose tile[i] <= chunk[i]

        chunk_shape = array.chunks
        # array.shape[i] >= chunk_shape[i] > 0

        # how many (full) chunk per axis
        chunk_count = [s // c for (s, c) in zip(array.shape, chunk_shape)]
        # chunk_count[i] > 0

        # select one random full chunk
        chunk_idx = [random.randrange(c-o)+o if c-o !=
                     0 else o for (c, o) in zip(chunk_count, self.overlap)]
        # chunk_count[i] > chunk_idx[i] >= overlap[i]

        # if overlap
        #   set a halfway before the chunk edge (to cross it)
        # else
        #   select a brand new point
        point = [i * c - t // 2 if o else i * c for (i, c, o, t)
                 in zip(chunk_idx, chunk_shape, self.overlap, tile)]
        # array.shape[i] > point[i] + tile[i]

        # get a tuple slice objects as index
        slice_tpl = tuple(slice(p, p+t) for (p, t) in zip(point, tile))

        # get data & simulate action
        data = array[slice_tpl]
        data.sum()


class Tile3DBenchmarkHDF5(Tile3DBenchmarkABC):
    "access an HDF5 multidimensional array with a random Nd slice"

    def get_array(self):
        return h5py.File(self.path, mode='r')['data']

    def info(self, _: Verbosity) -> str:
        file = h5py.File(self.path, mode='r')
        data: h5py.Dataset = file['data']

        info = f'HDF5 Companion: {self.path} - tile: {self.tile} - overlap: {self.overlap}'
        info += (f'\n\tshape = {data.shape}, chunks = {data.chunks}, '
                 f'compression = {data.compression}')

        file.close()
        return info


class Tile3DBenchmarkZarr(Tile3DBenchmarkABC):
    "access an Zarr multidimensional array with a random Nd slice"

    def get_array(self) -> zarr.Array:
        return zarr.open(self.path, mode='r')['0']

    def info(self, verbosity: Verbosity) -> str:
        data = self.get_array()

        info = f'Zarr Companion: {self.path} - tile: {self.tile} - overlap: {self.overlap}'

        if verbosity == Verbosity.VERBOSE:
            zarr_info = str(data.info)[:-1].split('\n')
            zarr_info = [item.split(':') for item in zarr_info]
            zarr_info = {l.strip(): r.strip() for (l, r) in zarr_info}

            info += (f'\n\tshape = {data.shape}, chunks = {data.chunks}, '
                     f'compressor = {data.compressor}, filters = {data.filters}')
            info += (f'\n\torder = {data.order}, size (mem) = {zarr_info["No. bytes"]}'
                     f', size (disk) {zarr_info["No. bytes stored"]}')

        return info


def run_benchmark(files: Iterable[str], number: int, repeat: int,
                  tile: Iterable[int], overlap: Iterable[int],
                  enable_gc: bool = False) -> None:
    "runs a HDF/Zarr benchmark on a list of files sequentially"

    for file in files:
        ext = pathlib.Path(file).suffix
        if ext == '.hdf5':
            Tile3DBenchmarkHDF5(file, tile, overlap).bench(
                Verbosity.VERBOSE, number=number, repeat=repeat, enable_gc=enable_gc)
        if ext == '.zarr':
            Tile3DBenchmarkZarr(file, tile, overlap).bench(
                Verbosity.VERBOSE, number=number, repeat=repeat, enable_gc=enable_gc)


# if __name__ == "__main__":
#     for overlap in (prefix + (0,) for prefix in itertools.product((0, 1), repeat=2)):
#         run_benchmark(['files/G1_6_top/profile.zarr'], number=1,
#                       repeat=1, tile=(128, 128, 1), overlap=overlap)
