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
    intensities = zarr.open_array(path + '/0', mode='r')
    mzs = zarr.open_array(path + '/labels/mzs/0', mode='r')

    if mzs.shape[-2:] == (1, 1):
        return ImzMLInfo(
            shape=intensities.shape[-2:],
            continuous_mode=True,
            path=path,
            band_size_min=mzs.shape[0],
            band_size_max=mzs.shape[0],
            mzs_min=np.min(mzs),
            mzs_max=np.max(mzs),
            mzs_precision=mzs.dtype,
            intensity_precision=intensities.dtype,
        )
    else:
        lengths = zarr.open_array(path + '/labels/lengths/0', mode='r')
        return ImzMLInfo(
            shape=intensities.shape,
            continuous_mode=False,
            path=path,
            band_size_min=np.min(lengths),
            band_size_max=np.max(lengths),
            mzs_min=np.min(mzs),
            mzs_max=np.max(mzs),
            mzs_precision=mzs.dtype,
            intensity_precision=intensities.dtype,
        )


def _zarr_info(file: str, verbosity: Verbosity, key: str) -> str:

    info = f'zarr from imzML file: {file} / {key}'

    if verbosity == Verbosity.VERBOSE:
        intensities = zarr.open_array(file + '/0', mode='r')
        mzs = zarr.open_array(file + '/labels/mzs/0', mode='r')

        is_continuous = mzs.shape[-2:] == (1, 1)

        info += f'\n\tbinary mode continuous: {is_continuous}'
        info += f"\n\tshape: {intensities.shape}"
        if not is_continuous:
            lengths = zarr.open_array(file + '/labels/lengths/0', mode='r')
            info += f' : band statistics: min={np.min(lengths)}, max={np.max(lengths)}'

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
        intensities = da.from_zarr(self.file, '/0')
        mzs = da.from_zarr(self.file, '/labels/mzs/0')

        x = random.randrange(intensities.shape[3])
        y = random.randrange(intensities.shape[2])

        if mzs.shape[-2:] == (1, 1):
            # suppose continuous
            np.dot(
                mzs[:, 0, 0, 0],
                intensities[:, 0, y, x]
            ).compute().tobytes()
        else:
            lengths = da.from_zarr(self.file, '/labels/lengths/0')
            length = lengths[0, 0, y, x].compute()[()]

            np.dot(
                mzs[:length, 0, y, x],
                intensities[:length, 0, y, x]
            ).compute().tobytes()

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
        intensities = zarr.open_array(self.file + '/0', mode='r')
        shape = intensities.shape[-2:]
        chunks = intensities.chunks[-2:]

        # should not specify a tile larger than the image's chunks
        if any(t > s for t, s in zip(tiles, chunks)):
            self.broken = True
            warnings.warn(f'{tiles=} too large for {chunks=}')
            return

        # there should be at least two full chunk
        if any(s // c < 2 for s, c in zip(shape, chunks)):
            self.broken = True
            warnings.warn(f'{chunks=} too large for {shape=}')
            return

    def task(self) -> None:
        intensities = da.from_zarr(self.file, '/0')
        mzs = da.from_zarr(self.file, '/labels/mzs/0')

        # continuous & processed mode have different shape -> take (x,y) only
        shape = intensities.shape[-2:]
        chunk_shape = intensities.chunksize[-2:]

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
        intensities = intensities[:, 0, slice_tpl[0], slice_tpl[1]]

        if mzs.shape[-2:] == (1, 1):
            intensities.sum(axis=0).compute().tobytes()
        else:
            lengths = da.from_zarr(self.file, '/labels/lengths/0')
            lengths = lengths[0, 0, slice_tpl[0], slice_tpl[1]]

            def load_window(int_val, len_val) -> float:
                return int_val[:len_val[()]].sum()

            da.apply_gufunc(load_window, '(i),()->()',
                            intensities, lengths,
                            axes=[(-3,), (), ()],
                            allow_rechunk=True,
                            vectorize=True).compute().tobytes()

    def info(self, verbosity: Verbosity) -> str:
        return _zarr_info(self.file, verbosity, f'sum access with tiles={self.tiles}')


class ZarrImzMLSumBenchmark(BenchmarkABC):
    "benchmark for imzML converted to zarr focussed on the sum of values accross all bands"

    def __init__(self, path: str, tiles: Tuple[int, int]) -> None:
        self.file = path
        self.tiles = tiles

        # make sure the file is big enough for the tiles
        intensities = zarr.open_array(self.file + '/0', mode='r')

        if any(t > s for t, s in zip(tiles, intensities.shape[-2:])):
            self.broken = True
            warnings.warn(f'{tiles=} too large for shape {intensities.shape}')

    def task(self) -> None:
        intensities = da.from_zarr(self.file, '/0')
        mzs = da.from_zarr(self.file, '/labels/mzs/0')

        point = [random.randrange(s-t)
                 for s, t in zip(intensities.shape[-2:], self.tiles)]
        slice_tpl = tuple(slice(p, p+t) for p, t in zip(point, self.tiles))

        intensities = intensities[:, 0, slice_tpl[0], slice_tpl[1]]

        if mzs.shape[-2:] == (1, 1):
            # suppose continuous
            intensities.sum(axis=0).compute().tobytes()
        else:
            lengths = da.from_zarr(self.file, '/labels/lengths/0')
            lengths = lengths[0, 0, slice_tpl[0], slice_tpl[1]]

            def load_window(int_val, len_val) -> float:
                return int_val[:len_val[()]].sum()

            da.apply_gufunc(load_window, '(i),()->()',
                            intensities, lengths,
                            axes=[(-3,), (), ()],
                            allow_rechunk=True,
                            vectorize=True).compute().tobytes()

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

        if any(t > s for t, s in zip(tiles, infos.shape[-2:])):
            self.broken = True
            warnings.warn(f'tiles {tiles} too large for shape {infos.shape}')

    def task(self) -> None:
        intensities = da.from_zarr(self.file, '/0')
        mzs = da.from_zarr(self.file, '/labels/mzs/0')

        mz_val = self.infos.mzs_min + random.random() * (self.infos.mzs_max -
                                                         self.infos.mzs_min)
        mz_tol = 0.1

        point = [random.randrange(s-t)
                 for s, t in zip(intensities.shape[-2:], self.tiles)]
        slice_tpl = tuple(slice(p, p+t) for p, t in zip(point, self.tiles))

        def search_processed(mz_band, int_band, len_val) -> float:
            # search low & high
            mz_band = mz_band[:len_val]
            low = np.searchsorted(mz_band, mz_val - mz_tol, side='left')
            high = 1 + np.searchsorted(mz_band, mz_val + mz_tol, side='right')
            return int_band[low:high].sum()

        if mzs.shape[-2:] == (1, 1):
            mz_band = mzs[:, 0, 0, 0]

            # search the unique mzs band
            low_i = da.searchsorted(mz_band, da.array([mz_val-mz_tol]),
                                    side='left')[0]
            high_i = 1 + da.searchsorted(mz_band, da.array([mz_val+mz_tol]),
                                         side='right')[0]

            img = intensities[
                low_i:high_i,  # m/Z
                0, slice_tpl[0], slice_tpl[1]  # z,y,x
            ].sum(axis=0)
        else:
            lengths = da.from_zarr(self.file, '/labels/lengths/0')

            mzs_window = mzs[:, 0, slice_tpl[0], slice_tpl[1]]
            int_window = intensities[:, 0, slice_tpl[0], slice_tpl[1]]
            lengths_window = lengths[0, 0, slice_tpl[0], slice_tpl[1]]

            img = da.apply_gufunc(search_processed, '(i),(i),()->()',
                                  mzs_window, int_window, lengths_window,
                                  axes=[(-3,), (-3,), (), ()],
                                  allow_rechunk=True,
                                  vectorize=True,
                                  )

        img.compute().tobytes()

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
        tiles = (20, 20)
        ZarrImzMLSearchBenchmark(file, tiles, infos).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)
        for overlap in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            ZarrImzMLOverlapSumBenchmark(file, tiles, overlap).bench(
                Verbosity.VERBOSE, enable_gc=args.gc, number=args.number,
                repeat=args.repeat)


if __name__ == "__main__":
    _main()
