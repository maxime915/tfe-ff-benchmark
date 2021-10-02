"hdf5 companion: benchmark HDF5 companion file for band access"

import random

import h5py

from .utils import parse_args
from .benchmark_abc import BenchmarkABC, Verbosity


class HDF5CompanionBenchmark(BenchmarkABC, re_str=r'.*profile\.hdf5'):
    "Benchmark for a profile HDF5 file: load spectral band"

    def __init__(self, file: str) -> None:
        self.file = file

    def task(self) -> None:
        'open an HDF5 file to read a random band then close the file'
        file = h5py.File(self.file, mode='r')

        # Y,X,C HDF5 dataset
        profile = file['data']

        y = random.randrange(profile.shape[0])
        x = random.randrange(profile.shape[1])

        band = profile[y, x, :]  # load it
        sum(band)  # do something with it

        file.close()

    def info(self, verbosity: Verbosity) -> None:
        file = h5py.File(self.file, mode='r')
        data = file['data']

        info = f'HDF5 Companion: {self.file}'

        if verbosity == Verbosity.VERBOSE:
            info += (f'\n\tshape = {data.shape}, chunks = {data.chunks}, '
                     f'compression = {data.compression}')

        file.close()
        return info


def _main():
    args = parse_args()

    for file in args.files:
        HDF5CompanionBenchmark(file).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)


if __name__ == "__main__":
    _main()

"""
HDF5CompanionBenchmark (duration averaged on 10000 iterations, repeated 3 times.)
results: [2.872e-04, 2.849e-04, 2.861e-04]: 2.849e-04 s to 2.872e-04 s, 2.861e-04 s ± 1.134e-06 s
HDF5 Companion: files/z-series/profile.hdf5
        shape = (167, 439, 5), chunks = None,  compression = None

HDF5CompanionBenchmark (duration averaged on 10000 iterations, repeated 3 times.)
results: [2.939e-04, 2.963e-04, 2.955e-04]: 2.939e-04 s to 2.963e-04 s, 2.952e-04 s ± 1.232e-06 s
HDF5 Companion: files/test-channel-image/profile.hdf5
        shape = (512, 512, 31), chunks = None,  compression = None
"""
