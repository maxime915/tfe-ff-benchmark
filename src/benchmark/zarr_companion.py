"hdf5 companion: benchmark Zarr companion file for band access"

import random

import zarr

from .utils import parse_args
from .benchmark_abc import BenchmarkABC, Verbosity


class ZarrCompanionBenchmark(BenchmarkABC, re_str=r'.*profile\.zarr'):
    "benchmark for profile converted to zarr"

    def __init__(self, path: str, *args, **kwargs) -> None:
        self.file = path

    def task(self) -> None:
        z = zarr.open(self.file, mode='r')

        # Y,X,C Zarr dataset
        profile = z['0']

        y = random.randrange(profile.shape[0])
        x = random.randrange(profile.shape[1])

        band = profile[y, x, :]  # load it
        sum(band)  # do something with it

    def info(self, verbosity: Verbosity) -> str:
        data = zarr.open(self.file, mode='r')['0']

        info = f'Zarr Companion: {self.file}'

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

    args = parse_args()

    for file in args.files:
        ZarrCompanionBenchmark(file).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)


if __name__ == "__main__":
    _main()

"""
ZarrCompanionBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [2.404e-04, 2.301e-04, 2.302e-04]: 2.301e-04 s to 2.404e-04 s, 2.336e-04 s ± 5.912e-06 s
Zarr Companion: files/z-series/profile.zarr
        shape = (167, 439, 5), chunks = (84, 439, 5), compressor = None, filters = None
        order = C, size (mem) = 366565 (358.0K), size (disk) 369004 (360.4K)

ZarrCompanionBenchmark (duration averaged on 1000 iterations, repeated 3 times.)
results: [3.583e-04, 3.521e-04, 3.527e-04]: 3.521e-04 s to 3.583e-04 s, 3.544e-04 s ± 3.372e-06 s
Zarr Companion: files/test-channel-image/profile.zarr
        shape = (512, 512, 31), chunks = (128, 128, 16), compressor = None, filters = None
        order = C, size (mem) = 16252928 (15.5M), size (disk) 16777463 (16.0M)
"""
