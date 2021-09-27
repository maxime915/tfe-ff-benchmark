"hdf5 companion: benchmark HDF5 companion file for band access"

import argparse
import random
import sys
import timeit
import typing

import h5py


# NOTE neither GC nor Context-Managers have an effect on small files


def print_info(file: str) -> None:
    ''
    group = h5py.File(file, mode='r')
    data = group['data']

    print(f'--- {file} --- ')
    print(f'shape = {data.shape}, chunk = {data.chunks}, compression = {data.compression}')

    group.close()


def benchmark_hdf5_companion(file: str, context_manager: bool) -> typing.Callable[[], None]:
    '''returns a callable that open an HDF5 file to read a random band and close
    the file'''

    def bench():
        'open an HDF5 file to read a random band then close the file'
        group = h5py.File(file, mode='r')

        # Y,X,C HDF5 dataset
        profile = group['data']

        y = random.randint(0, profile.shape[0] - 1)
        x = random.randint(0, profile.shape[1] - 1)

        band = profile[y, x, :] # load it
        sum(band) # do something with it

        group.close()
    
    def bench_with_context_manager():
        'open an HDF5 file to read a random band with context manager'
        with h5py.File(file, mode='r') as group:
            # Y,X,C HDF5 dataset
            profile = group['data']

            y = random.randint(0, profile.shape[0] - 1)
            x = random.randint(0, profile.shape[1] - 1)

            band = profile[y, x, :] # load it
            sum(band) # do something with it

    if context_manager:
        return bench_with_context_manager

    return bench

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-gc', dest='gc', action='store_false', help=
        "disable the garbage collector during measurements (default)")
    parser.add_argument('--gc', dest='gc', action='store_true', help=
        "enable the garbage collector during measurements")
    parser.set_defaults(gc=False)

    parser.add_argument('--no-context-manager', dest='cm', action='store_false', help=
        "don't use context manager (default)")
    parser.add_argument('--context-manager', dest='cm', action='store_true', help=
        "use a context manager")
    parser.set_defaults(cm=False)

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
    if args.cm:
        options.append('using context manager')
    if options:
        print('option(s): ' + ', '.join(options))
        
    for file in files:
        results = timeit.repeat(stmt='c()', setup=setup, repeat=args.repeat,
            number=args.number, globals={'c': benchmark_hdf5_companion(
                file=file,
                context_manager=args.cm
            )})
        print_info(file)
        print(f'{min(results):8.5f}s - {file}')
        
"""python -m src.benchmark.hdf5_companion files/z-series/profile.hdf5 files/test-channel-image/profile.hdf5                
benchmark.hdf5_companion: best of 3 (1000 iteration)
--- files/z-series/profile.hdf5 --- 
shape = (167, 439, 5), chunk = None, compression = None
 0.29461s - files/z-series/profile.hdf5
--- files/test-channel-image/profile.hdf5 --- 
shape = (512, 512, 31), chunk = None, compression = None
 0.30177s - files/test-channel-image/profile.hdf5
"""
