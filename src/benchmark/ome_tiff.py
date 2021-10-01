"pyramidal tiff: benchmark the internal pyramidal tiff representation"

import random
import typing

import tifffile

from .utils import parse_args_with_tile
from .benchmark_abc import BenchmarkABC, Verbosity


class OMETiffBenchmark(BenchmarkABC, re_str=r'.*\.ome\.tif(f)?'):
    "2D tile benchmark on OME-Tiff file"

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


def _main():
    args = parse_args_with_tile(ndim=2)

    for file in args.files:
        OMETiffBenchmark(file, tile=args.tile).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)


if __name__ == "__main__":
    _main()

"""
OMETiffBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [4.052e-04, 3.966e-04, 3.912e-04]: 3.912e-04 s to 4.052e-04 s, 3.977e-04 s ± 7.056e-06 s
OME-Tiff: files/z-series/z-series.ome.tif, tile: [128, 128]
        shape = 5 * (167, 439), chunks = (1, 439), chunked = (167, 1)

OMETiffBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [7.966e-04, 7.791e-04, 7.823e-04]: 7.791e-04 s to 7.966e-04 s, 7.860e-04 s ± 9.313e-06 s
OME-Tiff: files/test-channel-image/test_channel_image.ome.tiff, tile: [128, 128]
        shape = 31 * (512, 512), chunks = (1, 512), chunked = (512, 1)
"""
