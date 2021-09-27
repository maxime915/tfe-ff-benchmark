"hdf5 companion: benchmark Zarr companion file for band access"

import argparse
import random
import sys
import timeit
import typing

import zarr

def print_info(file: str) -> None:
    print(zarr.open(file, mode='r')[0].info)

def benchmark_zarr_companion(file: str) -> typing.Callable[[], None]:
    '''returns a callable that open an Zarr file to read a random band and close
    the file'''

    def bench():
        'open an Zarr file to read a random band then close the file'
        group = zarr.open(file, mode='r')

        # Y,X,C Zarr dataset
        profile = group['0']

        y = random.randrange(profile.shape[0])
        x = random.randrange(profile.shape[1])

        band = profile[y, x, :] # load it
        sum(band) # do something with it

    return bench

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-gc', dest='gc', action='store_false', help=
        "disable the garbage collector during measurements (default)")
    parser.add_argument('--gc', dest='gc', action='store_true', help=
        "enable the garbage collector during measurements")
    parser.set_defaults(gc=False)

    parser.add_argument('--number', type=int, default=1_000, help='see timeit')
    parser.add_argument('--repeat', type=int, default=3, help='see timeit')
    
    args, files = parser.parse_known_args()

    setup = 'import gc\ngc.enable()' if args.gc else 'pass'

    if len(files) <= 0:
        sys.exit("not enough file to benchmark")

    print(f"benchmark.hdf5_companion: best of {args.repeat} ({args.number} iteration)")
    
    options = []
    if args.gc:
        options.append('garbage collection enabled')
    if options:
        print('option(s): ' + ', '.join(options))
        
    for file in files:
        results = timeit.repeat(stmt='c()', setup=setup, repeat=args.repeat,
            number=args.number, globals={'c': benchmark_zarr_companion(
                file=file,
            )})
        
        print(f'{min(results):8.5f}s - {file}')
        
