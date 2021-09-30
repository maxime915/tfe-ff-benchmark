"pyramidal tiff: benchmark the internal pyramidal tiff representation"

import argparse
import random
import sys
import typing

import tifffile

from .benchmark_abc import BenchmarkABC, Verbosity


class OMETiffBenchmark(BenchmarkABC, re_str=r'.*\.ome\.tif(f)?'):

    def __init__(self, file: str, tile: typing.Tuple[int, int], **kwargs) -> None:
        super().__init__(file)

        self.file = file
        self.tile = tile

    def task(self) -> None:
        # Z pages of YX planes
        tif = tifffile.TiffFile(self.file)

        # get a random Z plane
        plane = random.choice(tif.pages)

        y = random.randrange(plane.shape[0] - self.tile[0])
        x = random.randrange(plane.shape[1] - self.tile[1])

        band = plane.asarray()[y:y+self.tile[0], x:x+self.tile[1]]  # load it
        band.sum()  # do something with it

        tif.close()

    def info(self, verbosity: Verbosity) -> str:
        tif = tifffile.TiffFile(self.file)
        page = tif.pages[0]

        info = f'OME-Tiff: {self.file}, tile: {self.tile}'

        if verbosity == verbosity.VERBOSE:
            info += (f'\n\tshape = {len(tif.pages)} * {page.shape}, '
                     f'chunks = {page.chunks}, chunked = {page.chunked}')

        tif.close()
        return info


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-gc', dest='gc', action='store_false',
                        help="disable the garbage collector during measurements (default)")
    parser.add_argument('--gc', dest='gc', action='store_true',
                        help="enable the garbage collector during measurements")
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
        OMETiffBenchmark(file, tile=args.tile).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)

"""
OMETiffBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [3.954e-04, 3.760e-04, 3.802e-04]: 3.760e-04 s to 3.954e-04 s, 3.839e-04 s ± 1.018e-05 s
OME-Tiff: files/z-series/z-series.ome.tif, tile: [128, 128]
        shape = 5 * (167, 439), chunk = (1, 439), chunked = (167, 1)
OMETiffBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [7.706e-04, 7.631e-04, 7.707e-04]: 7.631e-04 s to 7.707e-04 s, 7.681e-04 s ± 4.388e-06 s
OME-Tiff: files/test-channel-image/test_channel_image.ome.tiff, tile: [128, 128]
        shape = 31 * (512, 512), chunk = (1, 512), chunked = (512, 1)
"""
