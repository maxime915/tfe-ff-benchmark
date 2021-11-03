"imzml: benchmark imzML in tiled and spectral access"

import random

import numpy as np
import pyimzml.ImzMLParser

from .utils import parse_args
from .benchmark_abc import BenchmarkABC, Verbosity


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
    "benchmark for imzML focussed on the sum of values accross all bands"

    def __init__(self, path: str) -> None:
        self.file = path

    def task(self) -> None:
        parser = pyimzml.ImzMLParser.ImzMLParser(self.file)

        # flip x,y because images assume first dim as y
        shape = (parser.imzmldict['max count of pixels y'],
                 parser.imzmldict['max count of pixels x'])

        img = np.zeros(shape)

        # load data
        for idx, (x, y, _) in enumerate(parser.coordinates):
            # only read the intensity band
            parser.m.seek(parser.intensityOffsets[idx])

            img[y-1, x-1] = np.fromfile(parser.m, parser.intensityPrecision,
                                        parser.intensityLengths[idx]).sum()

        img.flatten()  # do something with it
        parser.m.close()

    def info(self, verbosity: Verbosity) -> str:
        return _imzml_info(self.file, verbosity, 'sum access')


def _main():

    args = parse_args()

    for file in args.files:
        ImzMLBandBenchmark(file).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)
        ImzMLSumBenchmark(file).bench(
            Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)


if __name__ == "__main__":
    _main()
