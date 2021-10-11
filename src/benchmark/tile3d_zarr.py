"tile3d_zarr: benchmark zarr file by loading a 3D tile"

import random
import sys
import typing

import zarr

from .benchmark_abc import BenchmarkABC, Verbosity
from .utils import basic_parser


class Tile3DZarrBenchmark(BenchmarkABC):
    "access a Zarr array with a random 3D slice"

    def __init__(self, path: str, tile: typing.List[int]) -> None:
        self.path = path
        self.tile = tile

    def task(self) -> None:
        array = zarr.open_group(self.path, mode='r')[0]

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

    def info(self, _: Verbosity) -> str:
        data = zarr.open_group(self.path, mode='r')['0']

        info = f'Zarr file: {self.path}, using tile = {self.tile}'

        zarr_info = str(data.info)[:-1].split('\n')
        zarr_info = [item.split(':') for item in zarr_info]
        zarr_info = {l.strip(): r.strip() for (l, r) in zarr_info}

        info += (f'\n\tshape = {data.shape}, chunks = {data.chunks}, '
                 f'compressor = {data.compressor}, filters = {data.filters}')
        info += (f'\n\torder = {data.order}, size (mem) = {zarr_info["No. bytes"]}'
                 f', size (disk) {zarr_info["No. bytes stored"]}')

        return info


def run_benchmark(files: typing.List[str], number: int, repeat: int,
                  tile: typing.Union[int, typing.Iterable[int]] = (32,), enable_gc: bool = False) -> None:

    for file in files:
        Tile3DZarrBenchmark(file, tile).bench(Verbosity.VERBOSE,
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
