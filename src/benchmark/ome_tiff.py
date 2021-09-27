"pyramidal tiff: benchmark the internal pyramidal tiff representation"

import argparse
import random
import sys
import timeit
import typing

import tifffile


def print_info(file: str) -> None:
    'print info for the first page of the tif file'
    
    tif = tifffile.TiffFile(file)
    page = tif.pages[0]

    print(f'--- {file} --- ')
    print(f'shape = {len(tif.pages)} * {page.shape}, chunk = {page.chunks}, chunked = {page.chunked}')

    tif.close()


def benchmark_ome_tiff(file: str, tile: typing.Tuple[int]) -> typing.Callable[[], None]:
    '''returns a callable that open an OME-TIFF file to read a random tile and close
    the file'''

    def bench():
        'open an OME-TIFF file to read a random tile then close the file'

        # Z pages of YX planes
        tif = tifffile.TiffFile(file)

        # get a random Z plane
        plane = random.choice(tif.pages)

        y = random.randint(0, plane.shape[0] - tile[0] - 1)
        x = random.randint(0, plane.shape[1] - tile[1] - 1)

        band = plane.asarray()[y:y+tile[0], x:x+tile[1]] # load it
        band.sum() # do something with it

        tif.close()


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
    parser.add_argument('--tile', nargs='*', default=['32'], help='tile size')
    
    args, files = parser.parse_known_args()

    if len(args.tile) not in [1, 2]:
        sys.exit(f'expected 0, 1 or 2 value for tile, found {len(args.tile)}')
    for i, tile in enumerate(args.tile):
        try:
            args.tile[i] = int(tile)
            if args.tile[i] <= 0:
                raise ValueError()
        except ValueError:
            sys.exit(f'invalid value {tile} for a tile')
    if len(args.tile) == 1:
        args.tile.append(args.tile[0])

    setup = 'import gc\ngc.enable()' if args.gc else 'pass'

    if len(files) <= 0:
        sys.exit("not enough file to benchmark")

    print(f"benchmark.hdf5_companion: best of {args.repeat} ({args.number} iteration) - tile = {args.tile}")
    
    options = []
    if args.gc:
        options.append('garbage collection enabled')
    if options:
        print('option(s): ' + ', '.join(options))
        
    for file in files:
        results = timeit.repeat(stmt='c()', setup=setup, repeat=args.repeat,
            number=args.number, globals={'c': benchmark_ome_tiff(
                file=file,
                tile=args.tile
            )})
        print_info(file)
        print(f'{min(results):8.5f}s')

