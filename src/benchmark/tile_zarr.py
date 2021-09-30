"tile zarr: read an OME-Tiff file converted to Zarr"

import argparse
import random
import sys
import typing

import zarr

from .benchmark_abc import BenchmarkABC, Verbosity

class TileZarrBenchmark(BenchmarkABC, re_str=r'.*\.zarr'):
    "2D Tile access to a Zarr image"

    def __init__(self, path: str, tile: typing.Tuple[int, ...], *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        self.file = file
        self.tile = tile
    
    def task(self) -> None:
        data = zarr.open(self.file, mode='r')[0]

        z = random.randrange(data.shape[0])
        y = random.randrange(data.shape[1] - self.tile[0])
        x = random.randrange(data.shape[2] - self.tile[1])

        band = data[z, y:y+self.tile[0], x:x+self.tile[1]] # load it
        band.sum() # do something with it
    
    def info(self, verbosity: Verbosity) -> str:
        data = zarr.open(self.file, mode='r')['0']

        info = f'Zarr file: {self.file}, tile: {self.tile}'

        if verbosity == Verbosity.VERBOSE:
            zarr_info = str(data.info)[:-1].split('\n')
            zarr_info = [item.split(':') for item in zarr_info]
            zarr_info = {l.strip() : r.strip() for (l, r) in zarr_info}

            info += (f'\n\tshape = {data.shape}, chunks = {data.chunks}, '
                     f'compressor = {data.compressor}, filters = {data.filters}')
            info += (f'\n\torder = {data.order}, size (mem) = {zarr_info["No. bytes"]}'
                     f', size (disk) {zarr_info["No. bytes stored"]}')

        return info


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-gc', dest='gc', action='store_false', help=
        "disable the garbage collector during measurements (default)")
    parser.add_argument('--gc', dest='gc', action='store_true', help=
        "enable the garbage collector during measurements")
    parser.set_defaults(gc=False)

    parser.add_argument('--number', type=int, default=1_000, help='see timeit')
    parser.add_argument('--repeat', type=int, default=3, help='see timeit')
    parser.add_argument('--tile', nargs='+', default=['32'], help='tile size')
    
    args, files = parser.parse_known_args()

    if len(args.tile) not in [1, 2]:
        sys.exit(f'expected 1 or 2 value for tile, found {len(args.tile)}')
    for i, tile in enumerate(args.tile):
        try:
            args.tile[i] = int(tile)
            if args.tile[i] <= 0:
                raise ValueError()
        except ValueError:
            sys.exit(f'invalid value {tile} for a tile')
    if len(args.tile) == 1:
        args.tile.append(args.tile[0])

    if len(files) <= 0:
        sys.exit("not enough file to benchmark")
        
    for file in files:
        TileZarrBenchmark(file, tile=args.tile).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)


"""
TileZarrBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [3.018e-04, 2.939e-04, 2.928e-04]: 2.928e-04 s to 3.018e-04 s, 2.962e-04 s ± 4.907e-06 s
Zarr file: files/test-channel-image/test_channel_image.zarr, tile: [32, 32]
        shape = (31, 512, 512), chunks = (8, 128, 256), compressor = None, filters = None
        order = C, size (mem) = 16252928 (15.5M), size (disk) 16777462 (16.0M)

TileZarrBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [2.417e-04, 2.395e-04, 2.376e-04]: 2.376e-04 s to 2.417e-04 s, 2.396e-04 s ± 2.063e-06 s
Zarr file: files/z-series/z-series.zarr, tile: [32, 32]
        shape = (5, 167, 439), chunks = (3, 167, 439), compressor = None, filters = None
        order = C, size (mem) = 366565 (358.0K), size (disk) 440123 (429.8K)
"""
