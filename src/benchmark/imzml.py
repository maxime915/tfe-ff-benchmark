"imzml: benchmark imzML in tiled and spectral access"

import random
from typing import NamedTuple, Tuple

import numpy as np
import pyimzml.ImzMLParser

from .utils import parse_args
from .benchmark_abc import BenchmarkABC, Verbosity


class ImzMLInfo(NamedTuple):
    'info of an imzML image (file independent)'
    shape: Tuple[int, int, int]  # x y z
    continuous_mode: bool
    path: str
    band_size_min: int  # for continuous mode, min and max are the same
    band_size_max: int
    mzs_min: float
    mzs_max: float
    intensity_precision: np.dtype
    mzs_precision: np.dtype


def imzml_path_to_info(path: str) -> ImzMLInfo:
    return parser_to_info(pyimzml.ImzMLParser.ImzMLParser(path, 'lxml'))


def _band_stat(mzs: np.ndarray) -> Tuple[int, float, float]:
    "build an ImzMLInfo object from a path to an imzml file"
    return len(mzs), mzs.min(), mzs.max()


def parser_to_info(parser: pyimzml.ImzMLParser.ImzMLParser) -> ImzMLInfo:
    "build an ImzMLInfo object from a pyimzml ImzMLParser"
    band_s_min, mzs_min, mzs_max = _band_stat(parser.getspectrum(0)[0])
    band_s_max = band_s_min
    band_s_sum = band_s_min

    limit = len(parser.coordinates)
    if 'continuous' in parser.metadata.file_description.param_by_name:
        band_s_sum *= len(parser.coordinates)
        limit = -1

    for i in range(1, limit):
        mzs, _ = parser.getspectrum(i)
        band_s, mz_min, mz_max = _band_stat(mzs)
        band_s_sum += len(mzs)
        if band_s < band_s_min:
            band_s_min = band_s
        elif band_s > band_s_max:
            band_s_max = band_s
        if mz_min < mzs_min:
            mzs_min = mz_min
        if mz_max > mzs_max:
            mzs_max = mz_max

    return ImzMLInfo(
        shape=(parser.imzmldict['max count of pixels x'],
               parser.imzmldict['max count of pixels y'],
               parser.imzmldict['max count of pixels z']),
        continuous_mode='continuous' in parser.metadata.file_description.param_by_name,
        path=parser.filename,
        band_size_min=band_s_min,
        band_size_max=band_s_max,
        mzs_min=mzs_min,
        mzs_max=mzs_max,
        intensity_precision=np.dtype(parser.intensityPrecision),
        mzs_precision=np.dtype(parser.mzPrecision),
    )


def _imzml_info(file: str, verbosity: Verbosity, key: str) -> str:

    info = f'ImzML file: {file} / {key}'

    if verbosity == Verbosity.VERBOSE:
        parser = pyimzml.ImzMLParser.ImzMLParser(file)

        is_continuous = 'continuous' in parser.metadata.file_description.param_by_name
        is_processed = 'processed' in parser.metadata.file_description.param_by_name

        assert(is_continuous != is_processed)

        info += f'\n\tbinary mode continuous: {is_continuous}'
        info += f"\n\tshape: ({parser.imzmldict['max count of pixels x']}, {parser.imzmldict['max count of pixels y']}"
        if is_continuous:
            info += f", {parser.mzLengths[0]})"
        else:
            min_mzs = np.min(parser.mzLengths)
            max_mzs = np.max(parser.mzLengths)
            info += f') : band statistics: {min_mzs=}, {max_mzs=}'

    return info


class ImzMLBandBenchmark(BenchmarkABC):
    "benchmark for imzML focussed on a single band"

    def __init__(self, path: str) -> None:
        self.file = path

    def task(self) -> None:
        parser = pyimzml.ImzMLParser.ImzMLParser(self.file)

        idx = random.randrange(len(parser.coordinates))
        mzs, intensities = parser.getspectrum(idx)  # load it

        np.dot(mzs, intensities)  # do something with it
        parser.m.close()

    def info(self, verbosity: Verbosity) -> str:
        return _imzml_info(self.file, verbosity, 'band access')


class ImzMLSumBenchmark(BenchmarkABC):
    "benchmark for imzML focussed on the sum of values accross all bands (TIC)"

    def __init__(self, path: str, tiles: Tuple[int, int]) -> None:
        self.file = path
        self.tiles = tiles

        # make sure the file is big enough for the tiles
        parser = pyimzml.ImzMLParser.ImzMLParser(self.file)
        shape = (parser.imzmldict['max count of pixels y'],
                 parser.imzmldict['max count of pixels x'])

        if any(t > s for t, s in zip(tiles, shape)):
            self.broken = True

    def task(self) -> None:
        parser = pyimzml.ImzMLParser.ImzMLParser(self.file)

        # flip x,y because images assume first dim as y
        shape = (parser.imzmldict['max count of pixels y'],
                 parser.imzmldict['max count of pixels x'])

        point = [random.randrange(shape[i] - self.tiles[i]) for i in range(2)]
        img = np.zeros(shape=self.tiles)

        # store mapping (x, y) -> idx for faster index in slice
        mapper = -np.ones(shape, dtype='i')
        for idx, (x, y, _) in enumerate(parser.coordinates):
            mapper[y-1, x-1] = idx

        # load data
        for y in range(point[0], point[0]+self.tiles[0]):
            for x in range(point[1], point[1]+self.tiles[1]):
                idx = mapper[y, x]
                if idx < 0:
                    continue
                parser.m.seek(parser.intensityOffsets[idx])
                img[y-point[0], x-point[1]] = np.fromfile(
                    parser.m, parser.intensityPrecision, parser.intensityLengths[idx]).sum()

        img.flatten()  # do something with it
        parser.m.close()

    def info(self, verbosity: Verbosity) -> str:
        return _imzml_info(self.file, verbosity, f'sum access with tiles={self.tiles}')


def _main():

    args = parse_args()

    for file in args.files:
        ImzMLBandBenchmark(file).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)
        ImzMLSumBenchmark(file, tiles=(100, 100)).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)


if __name__ == "__main__":
    _main()
