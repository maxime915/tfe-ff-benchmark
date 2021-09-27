"tile zarr: read an OME-Tiff file converted to Zarr"

import argparse
import random
import sys
import timeit
import typing

import zarr


def print_info(file: str) -> None:
    'print information about the file'
    print(zarr.open(file, mode='r')[0].info)


def benchmark_tile_zarr(file: str, tile: typing.Tuple[int]) -> typing.Callable[[], None]:
    '''returns a callable that open a Zarr file to read a random tile and close
    the file'''

    def bench():
        'open a Zarr file to read a random tile then close the file'

        data = zarr.open(file, mode='r')[0]

        z = random.randrange(data.shape[0])
        y = random.randrange(data.shape[1] - tile[0])
        x = random.randrange(data.shape[2] - tile[1])

        band = data[z, y:y+tile[0], x:x+tile[1]] # load it
        band.sum() # do something with it

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

    print(f"benchmark.hdf5_companion: best of {args.repeat} ({args.number} iteration), tile = {args.tile}")
    
    options = []
    if args.gc:
        options.append('garbage collection enabled')
    if options:
        print('option(s): ' + ', '.join(options))
        
    for file in files:
        results = timeit.repeat(stmt='c()', setup=setup, repeat=args.repeat,
            number=args.number, globals={'c': benchmark_tile_zarr(
                file=file,
                tile=args.tile
            )})
        print_info(file)
        print(f'{min(results):8.5f}s')

"""python -m src.benchmark.tile_zarr files/test-channel-image/test_channel_image.zarr files/z-series/z-series.zarr
benchmark.hdf5_companion: best of 3 (1000 iteration), tile = [32, 32]
Name               : /0
Type               : zarr.core.Array
Data type          : uint16
Shape              : (31, 512, 512)
Chunk shape        : (8, 128, 256)
Order              : C
Read-only          : True
Compressor         : None
Store type         : zarr.storage.DirectoryStore
No. bytes          : 16252928 (15.5M)
No. bytes stored   : 16777462 (16.0M)
Storage ratio      : 1.0
Chunks initialized : 32/32

 0.29240s
Name               : /0
Type               : zarr.core.Array
Data type          : int8
Shape              : (5, 167, 439)
Chunk shape        : (3, 167, 439)
Order              : C
Read-only          : True
Compressor         : None
Store type         : zarr.storage.DirectoryStore
No. bytes          : 366565 (358.0K)
No. bytes stored   : 440123 (429.8K)
Storage ratio      : 0.8
Chunks initialized : 2/2

 0.23037s"""