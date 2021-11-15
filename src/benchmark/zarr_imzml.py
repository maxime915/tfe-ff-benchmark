"zarr_imzml: benchmark imzML converted to zarr in tiled and spectral access"

import random
from typing import Tuple
import warnings

import dask.array as da
import numpy as np
import zarr
from zarr.util import human_readable_size

from .utils import parse_args
from .benchmark_abc import BenchmarkABC, Verbosity
from .imzml import ImzMLInfo


def zarr_imzml_path_to_info(path: str) -> ImzMLInfo:
    intensities = zarr.open_array(path + '/intensities', mode='r')
    mzs = zarr.open_array(path + '/mzs', mode='r')

    if len(mzs.shape) == 1:
        return ImzMLInfo(
            shape=intensities.shape[:-1],
            continuous_mode=True,
            path=path,
            band_size_min=mzs.shape[0],
            band_size_max=mzs.shape[0],
            mzs_min=np.min(mzs),
            mzs_max=np.max(mzs),
        )
    else:
        return ImzMLInfo(
            shape=intensities.shape,
            continuous_mode=False,
            path=path,
            band_size_min=np.vectorize(len)(mzs).min(),
            band_size_max=np.vectorize(len)(mzs).max(),
            mzs_min=np.vectorize(np.min)(mzs).min(),
            mzs_max=np.vectorize(np.max)(mzs).max(),
            mzs_precision=mzs.dtype,
            intensity_precision=intensities.dtype,
        )


def _zarr_info(file: str, verbosity: Verbosity, key: str) -> str:

    info = f'zarr from imzML file: {file} / {key}'

    if verbosity == Verbosity.VERBOSE:
        intensities = zarr.open_array(file + '/intensities', mode='r')
        mzs = zarr.open_array(file + '/mzs', mode='r')

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


class ZarrImzMLOverlapSumBenchmark(BenchmarkABC):
    """benchmark for imzML converted to zarr focussed on the sum of values
    accross all bands, loading on chunks border or inside full chunks

    there is  img.shape[i], img.chunksize[i], tiles[i], overlap[i]

    - img.chunksize[i] <= img.shape[i]  (enforced by zarr iff all chunks are full size)
    - chunk_count[i] = img.shape[i] // img.shape[i]
    - we need chunk_count[i] > 1 to make sure there are at least two full chunks
    - we need tiles[i] <= img.chunksize[i]  (otherwise we would span too many chunks)

    generation:

    - it we should overlap for axis i
        - choose a point on a random 'right' border at axis i
        - offset the point by tiles[i]/2 to the 'left'
    - otherwise
        - choose a point on a random 'right' border at axis i

    """

    def __init__(self, path: str, tiles: Tuple[int, int], overlap: Tuple[int, int]) -> None:
        self.file = path
        self.tiles = tiles
        self.overlap = overlap

        # make sure the file is big enough for the tiles
        intensities = zarr.open_array(self.file + '/intensities', mode='r')

        # should not specify a tile larger than the image's chunks
        if any(t > s for t, s in zip(tiles, intensities.chunks[:2])):
            self.broken = True
            warnings.warn(f'tiles {tiles} too large for chunks {intensities.chunks[:2]}')
            return

        # there should be at least two full chunk
        if any(s // c < 2 for s, c in zip(intensities.shape[:2], intensities.chunks[:2])):
            self.broken = True
            warnings.warn(f'chunks {intensities.chunks[:2]} too large for shape {intensities.shape[:2]}')
            return

    def task(self) -> None:
        intensities = da.from_zarr(self.file, '/intensities')
        mzs = da.from_zarr(self.file, '/mzs')

        # continuous & processed mode have different shape -> take (x,y) only
        shape = intensities.shape[:2]
        chunk_shape = intensities.chunksize[:2]

        # how many (full) chunk per axis
        chunk_count = [s // c for s, c in zip(shape, chunk_shape)]

        # select one random full chunk to use as overlap border
        chunk_idx = [random.randrange(c-o)+o if c-o !=
                     0 else o for c, o in zip(chunk_count, self.overlap)]

        # if overlap
        #   set a halfway before the chunk edge (to make sure it is crossed)
        # else
        #   select the beginning of the chunk
        point = [i * c - t // 2 if o else i * c for i, c, o,
                 t in zip(chunk_idx, chunk_shape, self.overlap, self.tiles)]

        # get a tuple slice object as index
        slice_tpl = tuple(slice(p, p+t) for p, t in zip(point, self.tiles))

        # pre-load tile
        intensities = intensities[slice_tpl]

        if len(mzs.shape) == 1:
            intensities.sum(axis=-1).compute().flatten()
        else:
            # no need to compute()
            np.vectorize(np.sum)(intensities).flatten()

    def info(self, verbosity: Verbosity) -> str:
        return _zarr_info(self.file, verbosity, f'sum access with tiles={self.tiles}')


class ZarrImzMLSumBenchmark(BenchmarkABC):
    "benchmark for imzML converted to zarr focussed on the sum of values accross all bands"

    def __init__(self, path: str, tiles: Tuple[int, int]) -> None:
        self.file = path
        self.tiles = tiles

        # make sure the file is big enough for the tiles
        intensities = zarr.open_array(self.file + '/intensities', mode='r')

        if any(t > s for t, s in zip(tiles, intensities.shape[:2])):
            self.broken = True
            warnings.warn(f'tiles {tiles} too large for shape {intensities.shape}')

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


class ZarrImzMLSearchBenchmark(BenchmarkABC):
    """benchmark for imzML converted to zarr focussed on an approximate slice
    of value close to m/z"""

    def __init__(self, path: str, tiles: Tuple[int, int],
                 infos: ImzMLInfo) -> None:
        self.file = path
        self.tiles = tiles
        self.infos = infos

        if any(t > s for t, s in zip(tiles, infos.shape[:2])):
            self.broken = True
            warnings.warn(f'tiles {tiles} too large for shape {infos.shape}')

    def task(self) -> None:
        intensities = da.from_zarr(self.file, '/intensities')
        mzs = da.from_zarr(self.file, '/mzs')

        mz_val = self.infos.mzs_min + random.random() * (self.infos.mzs_max -
                                                         self.infos.mzs_min)
        mz_tol = 0.1

        point = [random.randrange(
            intensities.shape[i] - self.tiles[i]) for i in range(2)]
        slice_tpl = tuple(slice(p, p+t) for p, t in zip(point, self.tiles))

        @da.as_gufunc(signature='(),()->()', output_dtypes=float, vectorize=True)
        def search_processed_band(mz_band: np.ndarray, int_band: np.ndarray) -> float:

            low = np.searchsorted(mz_band, mz_val - mz_tol, side='left')
            high = 1 + np.searchsorted(mz_band, mz_val + mz_tol, side='right')

            # reduction of the intensity band (wbt empty bands?)
            return int_band[low:high].sum()

        def do_continuous_image() -> da.Array:
            # search the unique mzs band
            low = da.searchsorted(mzs, da.array([mz_val-mz_tol]),
                                  side='left')[0]
            high = 1 + da.searchsorted(mzs, da.array([mz_val+mz_tol]),
                                       side='right')[0]

            return intensities[..., low:high].sum(axis=-1)

        # grab sub window
        intensities = intensities[slice_tpl]
        if len(mzs.shape) > 1:
            mzs = mzs[slice_tpl]  # sliceable only in processed mode
            img = search_processed_band(mzs, intensities).compute()
        else:
            img = do_continuous_image()
        img.flatten()

    def info(self, verbosity: Verbosity) -> str:
        return _zarr_info(self.file, verbosity, f'sum access with tiles={self.tiles}')


def _main():

    args = parse_args()

    for file in args.files:
        ZarrImzMLBandBenchmark(file).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)
        ZarrImzMLSumBenchmark(file, tiles=(10, 10)).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)
        infos = zarr_imzml_path_to_info(file)
        tiles = (32, 32)
        ZarrImzMLSearchBenchmark(file, tiles, infos).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)


if __name__ == "__main__":
    _main()
