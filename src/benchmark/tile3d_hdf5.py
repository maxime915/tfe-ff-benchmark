"tile3d_hdf5: benchmark HDF5 file by loading a 3D tile"

import random
import sys
import typing

import h5py

from .benchmark_abc import BenchmarkABC, Verbosity
from .utils import basic_parser


class Tile3DHDF5Benchmark(BenchmarkABC):
    "access an HDF5 dataset with a random 3D slice"

    def __init__(self, path: str, tile: typing.List[int]) -> None:
        self.path = path
        self.tile = tile

    def task(self) -> None:
        file = h5py.File(self.path, mode='r')
        array = file['data']

        # re-compute slice based on array shape
        tile = (t if t > 0 else s // -t for (s, t)
                in zip(array.shape, self.tile))

        # top left corner of the tile
        point = (random.randrange(s - t) if s !=
                 t else 0 for (s, t) in zip(array.shape, tile))

        # get data
        slice_lst = tuple(slice(p, p+t) for (p, t) in zip(point, tile))
        data = array[slice_lst]

        # simulate action
        data.sum()
        file.close()

    def info(self, _: Verbosity) -> str:
        file = h5py.File(self.path, mode='r')
        data = file['data']

        info = f'HDF5 Companion: {self.path}, using tile = {self.tile}'
        info += (f'\n\tshape = {data.shape}, chunks = {data.chunks}, '
                 f'compression = {data.compression}')

        file.close()
        return info


def run_benchmark(files: typing.List[str], number: int, repeat: int,
                  tile: typing.Union[int, typing.Iterable[int]] = (32,), enable_gc: bool = False) -> None:

    for file in files:
        Tile3DHDF5Benchmark(file, tile).bench(Verbosity.VERBOSE,
            number=number, repeat=repeat, enable_gc=enable_gc)


def _main():
    parser = basic_parser()

    parser.add_argument('--tile', nargs='+', default=[128, 128, 128], type=int,
                        help="size of the tile to load. if tile[i] < 0 then shape[i] // (-tile[i]) is used.")

    args = parser.parse_args()

    tile = args.tile
    if len(tile) == 1:
        tile = 3 * tile
    elif len(tile) > 3:
        sys.exit(
            f'invalid tile size {tile}, must not contain more than 3 values')
    elif len(tile) == 2:
        tile.append(1)

    for t in tile:
        if t == 0:
            sys.exit(f'invalid tile size (0) in {tile}')

    run_benchmark(args.files, args.number, args.repeat, tile, args.gc)


if __name__ == '__main__':
    _main()
