'companion to zarr : convert an HDF5 companion to Zarr'

import argparse
import sys
import timeit
import typing

import h5py
import numpy
import zarr


def converter(hdf5_path: str, zarr_path: str, **kwargs) -> typing.Callable[[], None]:
    "returns a function that does the convertion"

    def converter_fun() -> None:
        hdf5_file = h5py.File(hdf5_path, mode='r')
        zarr_file = zarr.open_group(zarr_path, mode='w')

        zarr.copy(hdf5_file['data'], zarr_file, name='0', **kwargs)
        hdf5_file.close()

        # (partial) OME-Zarr attributes
        zarr_file.attrs['multiscales'] = {
            'version': '0.3',
            'name': zarr_path,
            'datasets': [{'path': '0'}],
            'axes': ['t', 'c', 'z', 'y', 'x'],
            'type': 'conversion from HDF5 profile',
            'metadata': {'original': hdf5_path},
        }

    return converter_fun


def convert(hdf5_path: str, zarr_path: str, **kwargs) -> None:
    "convert an HDF5 profile file to a zarr equivalent"
    converter(hdf5_path, zarr_path, **kwargs)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('hdf5', type=str, help='input HDF5 file')
    parser.add_argument('zarr', nargs='?', type=str, default=None,
                        help='output path, defaults to random unique suffix')

    # chunk
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument('--auto-chunk', dest='auto-chunk', action='store_true',
                             help='automatic chunking for array storage (default option)')
    chunk_group.add_argument('--no-chunk', dest='auto-chunk', action='store_false',
                             help='disable chunking for array storage')
    chunk_group.add_argument('--chunks', type=int, nargs='+', default=[],
                             help='Chunk shape : list of positive integers (-1 means equal to array shape)')
    parser.set_defaults(auto_chunk=True)

    # C / Fortran array order
    parser.add_argument('--order', default='C', choices=['C', 'F'],
                        help='Memory layout to be used within each chunk (C vs Fortran)')

    # metadata chaching
    parser.add_argument('--cache-metadata', default=True,
                        choices=[True, False], help='If True, array configuration metadata '
                        'will be cached for the lifetime of the object. If False, array '
                        'metadata will be reloaded prior to all data access and modification '
                        'operations (may incur overhead depending on storage and data access pattern).')

    # Compressor -> default, none, or some evaluated object
    parser.add_argument('--compressor', type=str, default="default", help='either'
                        + ' "default", "none" or some python code to construct'
                        + ' the compressor object')

    parser.add_argument('--benchmark', action='store_true',
                        help="time conversion")

    args = parser.parse_args()

    if args.zarr is None:
        # get unique new filename
        import uuid
        args.zarr = args.hdf5[:-5] + str(uuid.uuid4()) + '.zarr'

    chunks = args.auto_chunk
    if args.chunks:
        chunks = args.chunks
        for i, c in enumerate(chunks):
            try:
                chunks[i] = int(c)
                if chunks[i] == -1:
                    pass  # will be set to shape[i]
                elif chunks[i] < 1:
                    raise ValueError(f'"{c}" is not a valid chunk size')
            except ValueError as e:
                sys.exit(f"unable to parse chunk option: {args.chunk}")

    # compressor : None, default or some evaluated object
    if args.compressor == 'none':
        args.compressor = None
    elif args.compressor != 'default':
        # Zlib, GZip, BZ2, LZMA?, Blosc, Zstd, lz4, ZFPY, and a lot of others...
        import numcodecs
        import codecs
        try:
            args.compressor = eval(args.compressor)
        except Exception as e:
            print(f'unable to evaluate compressor\n\n{e}')

    if args.benchmark:
        results = timeit.Timer(converter(args.hdf5, args.zarr, chunks=chunks, compressor=args.compressor,
                               cache_metadata=args.cache_metadata, order=args.order)).repeat(3, number=10)
        results = numpy.array(results) / 10
        res = "[" + ", ".join([f'{v:5.3e}' for v in results]) + "]"
        print((f'results: {res}: {results.min():5.3e} s to {results.max():5.3e} s, '
               f'{results.mean():5.3e} s ± {results.std(ddof=1):5.3e} s'))
    else:
        convert(args.hdf5, args.zarr, chunks=chunks, compressor=args.compressor,
                cache_metadata=args.cache_metadata, order=args.order)

    print(zarr.open_group(args.zarr, mode='r')['0'].info)
