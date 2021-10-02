"pyramidal tiff: benchmark the internal pyramidal tiff representation"

import os
import random
import typing

import tifffile

from .benchmark_abc import BenchmarkABC, Verbosity
from .utils import parse_args_with_tile

_VALID_SUFFIX_S = ('_C0_Z0_T0.tif', '_C0_Z0_T0_pyr.tif')


class PyramidalTiffBenchmark(BenchmarkABC, re_str=r'.*_pyr\.tif(f)?'):
    '2D Benchmark opening random tile in the image'

    def __init__(self, file: str, tile: typing.Tuple[int, int]) -> None:
        base_name = None
        for suffix in _VALID_SUFFIX_S:
            if file[-len(suffix):] == suffix:
                base_name = file[:-len(suffix)] + '_C0_Z'
                break

        if base_name is None:
            raise ValueError(
                f'INVALID file: {file} should end in any({_VALID_SUFFIX_S})')

        self.file = file
        self.base_name = base_name
        self.ext = suffix[6:]
        self.max_z = len(os.listdir(os.path.dirname(base_name))) // 2
        self.tile = tile

    def task(self) -> None:
        file = self.base_name + str(random.randrange(self.max_z)) + self.ext

        # Z pages of YX planes
        tif = tifffile.TiffFile(file)

        # get a random Z plane
        plane = tif.pages[0]

        y = random.randrange(plane.shape[0] - self.tile[0])
        x = random.randrange(plane.shape[1] - self.tile[1])

        data = plane.asarray()[y:y+self.tile[0], x:x+self.tile[1]]  # load it
        data.sum()  # do something with it

        tif.close()

    def info(self, verbosity: Verbosity) -> str:

        tif = tifffile.TiffFile(self.file)
        page = tif.pages[0]

        info = f'Pyramidal Tiff: {self.file}, tile: {self.tile}'

        if verbosity == verbosity.VERBOSE:
            # FIXME ?
            # len(tif.pages) trigger Warning 'TiffPages: invalid page offset' non _pyr files
            info += (f'\n\tshape = {len(tif.pages)} * {page.shape}, '
                     f'chunks = {page.chunks}, chunked = {page.chunked}')

        tif.close()
        return info


def _main():
    args = parse_args_with_tile(ndim=2)

    for file in args.files:
        PyramidalTiffBenchmark(file, args.tile).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)


if __name__ == "__main__":
    _main()

"""
PyramidalTiffBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [4.224e-04, 4.057e-04, 4.099e-04]: 4.057e-04 s to 4.224e-04 s, 4.127e-04 s ± 8.702e-06 s
TiffPages: invalid page offset 9239
Pyramidal Tiff: files/z-series/conversion/z-series_C0_Z0_T0.tif, tile: [32, 32]
        shape = 1 * (167, 439), chunks = (167, 256), chunked = (1, 2)

PyramidalTiffBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [4.643e-03, 3.509e-03, 3.118e-03]: 3.118e-03 s to 4.643e-03 s, 3.757e-03 s ± 7.924e-04 s
TiffPages: invalid page offset 609098
Pyramidal Tiff: files/test-channel-image/conversion/test_channel_image_C0_Z0_T0.tif, tile: [32, 32]
        shape = 1 * (512, 512), chunks = (256, 256), chunked = (2, 2)

PyramidalTiffBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [5.133e-04, 4.968e-04, 4.952e-04]: 4.952e-04 s to 5.133e-04 s, 5.018e-04 s ± 1.003e-05 s
Pyramidal Tiff: files/z-series/conversion/z-series_C0_Z0_T0_pyr.tif, tile: [32, 32]
        shape = 2 * (167, 439), chunks = (256, 256), chunked = (1, 2)

PyramidalTiffBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [3.276e-03, 3.431e-03, 3.416e-03]: 3.276e-03 s to 3.431e-03 s, 3.374e-03 s ± 8.559e-05 s
Pyramidal Tiff: files/test-channel-image/conversion/test_channel_image_C0_Z0_T0_pyr.tif, tile: [32, 32]
        shape = 2 * (512, 512), chunks = (256, 256), chunked = (2, 2)
"""
