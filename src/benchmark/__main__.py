"benchmark: "

import argparse
import re
import sys

# register all benchmark classes
from .hdf5_companion import HDF5CompanionBenchmark
from .ome_tiff import OMETiffBenchmark
from .pyramidal_tiff import PyramidalTiffBenchmark
from .tile_zarr import TileZarrBenchmark
from .zarr_companion import ZarrCompanionBenchmark

# import base
from .benchmark_abc import BenchmarkABC, Verbosity

parser = argparse.ArgumentParser()

parser.add_argument('files', type=str, nargs='+', help='filename')

parser.add_argument('--no-gc', dest='gc', action='store_false',
                    help="disable the garbage collector during measurements (default)")
parser.add_argument('--gc', dest='gc', action='store_true',
                    help="enable the garbage collector during measurements")
parser.set_defaults(gc=False)

parser.add_argument('--number', type=int, default=1_000, help='see timeit')
parser.add_argument('--repeat', type=int, default=3, help='see timeit')

args, others = parser.parse_known_args()
kwargs = dict([value.split('=') for value in others])

for file in args.files:
    found = False
    for re_str, benchmark_cls in BenchmarkABC.get_ordered_mapper().items():

        if re.match(re_str, file) is not None:
            found = True
            benchmark_cls(file, **kwargs).bench(
                Verbosity.VERBOSE, enable_gc=args.gc, number=args.number, repeat=args.repeat)
    
    if not found:
        sys.exit(f"unable to find a Benchmark for '{file}'")