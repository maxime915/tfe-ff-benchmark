"pyramidal tiff: benchmark the internal pyramidal tiff representation"

import argparse
import os
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
    # FIXME len(tif.pages) trigger a warning 'TiffPages: invalid page offset'
    print(f'shape = {len(tif.pages)} * {page.shape}, chunk = {page.chunks}, chunked = {page.chunked}')

    tif.close()
    

def benchmark_pyramidal_tiff(base_name: str, ext: str, tile: typing.Tuple[int]) -> typing.Callable[[], None]:
    '''returns a callable that open an HDF5 file to read a random tile and close
    the file
    
    file = base_name + str(z) + ext with increasing z values'''

    # this would be in a database access with the filename, not accounted for
    max_z = len(os.listdir(os.path.dirname(base_name))) // 2
    
    def bench():
        'open an HDF5 file to read a random tile then close the file'

        file = base_name + str(random.randrange(max_z)) + ext

        # Z pages of YX planes
        tif = tifffile.TiffFile(file)

        # get a random Z plane
        plane = tif.pages[0]

        y = random.randrange(plane.shape[0] - tile[0])
        x = random.randrange(plane.shape[1] - tile[1])

        data = plane.asarray()[y:y+tile[0], x:x+tile[1]] # load it
        data.sum() # do something with it

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

    print(f"benchmark.hdf5_companion: best of {args.repeat} ({args.number} iteration) tile = {args.tile}")
    
    options = []
    if args.gc:
        options.append('garbage collection enabled')
    if options:
        print('option(s): ' + ', '.join(options))
        
    for file in files:
        # file is expected to end in C0_Z0_T0.tif or C0_Z0_T0_pyr.tif
        suffix_list = ['_C0_Z0_T0.tif', '_C0_Z0_T0_pyr.tif']

        base_name = None
        for suffix in suffix_list:
            if file[-len(suffix):] == suffix:
                base_name = file[:-len(suffix)] + '_C0_Z'
                break
        
        if base_name is None:
            print(f'INVALID file: {file} should end in any({suffix_list})')
            continue

        results = timeit.repeat(stmt='c()', setup=setup, repeat=args.repeat,
            number=args.number, globals={'c': benchmark_pyramidal_tiff(
                base_name=base_name,
                ext=suffix[6:],
                tile=args.tile,
            )})
        print_info(file)
        print(f'{min(results):8.5f}s')
