"tile zarr: read an OME-Tiff file converted to Zarr"

import random
import typing

import zarr

from .benchmark_abc import BenchmarkABC, Verbosity
from .utils import parse_args_with_tile


class TileZarrBenchmark(BenchmarkABC, re_str=r'.*\.zarr'):
    "2D Tile access to a Zarr image"

    def __init__(self, file: str, tile: typing.Tuple[int, ...]) -> None:
        self.file = file
        self.tile = tile

    def task(self) -> None:
        data = zarr.open(self.file, mode='r')[0]

        z = random.randrange(data.shape[0])
        y = random.randrange(data.shape[1] - self.tile[0])
        x = random.randrange(data.shape[2] - self.tile[1])

        band = data[z, y:y+self.tile[0], x:x+self.tile[1]]  # load it
        band.sum()  # do something with it

    def info(self, verbosity: Verbosity) -> str:
        data = zarr.open(self.file, mode='r')['0']

        info = f'Zarr file: {self.file}, tile: {self.tile}'

        if verbosity == Verbosity.VERBOSE:
            zarr_info = str(data.info)[:-1].split('\n')
            zarr_info = [item.split(':') for item in zarr_info]
            zarr_info = {l.strip(): r.strip() for (l, r) in zarr_info}

            info += (f'\n\tshape = {data.shape}, chunks = {data.chunks}, '
                     f'compressor = {data.compressor}, filters = {data.filters}')
            info += (f'\n\torder = {data.order}, size (mem) = {zarr_info["No. bytes"]}'
                     f', size (disk) {zarr_info["No. bytes stored"]}')

        return info


def _main():
    args = parse_args_with_tile(ndim=2)

    for file in args.files:
        TileZarrBenchmark(file, tile=args.tile).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)


if __name__ == "__main__":
    _main()

"""
TileZarrBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [3.200e-04, 3.162e-04, 3.264e-04]: 3.162e-04 s to 3.264e-04 s, 3.209e-04 s ± 5.169e-06 s
Zarr file: files/test-channel-image/test_channel_image.zarr, tile: [32, 32]
        shape = (31, 512, 512), chunks = (8, 128, 256), compressor = None, filters = None
        order = C, size (mem) = 16252928 (15.5M), size (disk) 16777462 (16.0M)

TileZarrBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [2.603e-04, 2.463e-04, 2.502e-04]: 2.463e-04 s to 2.603e-04 s, 2.523e-04 s ± 7.192e-06 s
Zarr file: files/z-series/z-series.zarr, tile: [32, 32]
        shape = (5, 167, 439), chunks = (3, 167, 439), compressor = None, filters = None
        order = C, size (mem) = 366565 (358.0K), size (disk) 440123 (429.8K)
"""
