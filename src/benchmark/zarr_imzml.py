"zarr_imzml: benchmark imzML converted to zarr in tiled and spectral access"

import random
from typing import Tuple

import dask.array as da
import numpy as np
import zarr
from zarr.util import human_readable_size

from .utils import parse_args
from .benchmark_abc import BenchmarkABC, Verbosity


def _zarr_info(file: str, verbosity: Verbosity, key: str) -> str:

    info = f'zarr from imzML file: {file} / {key}'

    if verbosity == Verbosity.VERBOSE:
        group = zarr.open(file, mode='r')
        intensities = group.intensities
        mzs = group.mzs

        is_continuous = len(mzs.shape) == 1

        info += f'\n\tbinary mode continuous: {is_continuous}'
        info += f"\n\tshape: {intensities.shape}"
        if not is_continuous:
            lengths = np.vectorize(len)(intensities)
            info += f' : band statistics: min={lengths.min()}, max={lengths.max()}'

        # storage info
        info += f'\n\t[intensities] No. bytes: {intensities.nbytes} ({human_readable_size(intensities.nbytes)})'
        info += f'\n\t[intensities] No. bytes stored: {intensities.nbytes_stored} ({human_readable_size(intensities.nbytes_stored)})'

        info += f'\n\t[mzs] No. bytes: {mzs.nbytes} ({human_readable_size(mzs.nbytes)})'
        info += f'\n\t[mzs] No. bytes stored: {mzs.nbytes_stored} ({human_readable_size(mzs.nbytes_stored)})'

    return info


class ZarrImzMLBandBenchmark(BenchmarkABC):
    "benchmark for imzML converted to zarr focussed on a single band"

    def __init__(self, path: str) -> None:
        self.file = path

    def task(self) -> None:
        intensities = da.from_zarr(self.file, '/intensities')
        mzs = da.from_zarr(self.file, '/mzs')

        x = random.randrange(intensities.shape[0])
        y = random.randrange(intensities.shape[1])

        intensity_band = intensities[x, y].compute()

        if len(mzs.shape) == 1:
            mzs_band = mzs.compute()
        else:
            mzs_band = mzs[x, y].compute()

        np.dot(mzs_band, intensity_band)

    def info(self, verbosity: Verbosity) -> str:
        return _zarr_info(self.file, verbosity, 'band access')


class ZarrImzMLSumBenchmark(BenchmarkABC):
    "benchmark for imzML converted to zarr focussed on the sum of values accross all bands"

    def __init__(self, path: str, tiles: Tuple[int, int]) -> None:
        self.file = path
        self.tiles = tiles

        # make sure the file is big enough for the tiles
        intensities = da.from_zarr(self.file, '/intensities')

        if any(t > s for t, s in zip(tiles, intensities.shape[:2])):
            self.broken = True

    def task(self) -> None:
        intensities = da.from_zarr(self.file, '/intensities')
        mzs = da.from_zarr(self.file, '/mzs')

        point = [random.randrange(
            intensities.shape[i] - self.tiles[i]) for i in range(2)]
        slice_tpl = tuple(slice(p, p+t) for p, t in zip(point, self.tiles))

        intensities = intensities[slice_tpl]

        if len(mzs.shape) == 1:
            intensities.sum(axis=-1).compute().flatten()
        else:
            # no need to compute()
            np.vectorize(np.sum)(intensities).flatten()

    def info(self, verbosity: Verbosity) -> str:
        return _zarr_info(self.file, verbosity, f'sum access with tiles={self.tiles}')


def _main():

    args = parse_args()

    for file in args.files:
        ZarrImzMLBandBenchmark(file).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)
        ZarrImzMLSumBenchmark(file, tiles=(10, 10)).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)


if __name__ == "__main__":
    _main()
