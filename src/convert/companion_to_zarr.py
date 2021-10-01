'companion to zarr : convert an HDF5 companion to Zarr'

import argparse

import h5py
import zarr


def convert(h_file: str, z_file: str, **kwargs) -> None:
    h5 = h5py.File(h_file, mode='r')
    z = zarr.open_group(z_file, mode='w')

    zarr.copy(h5['data'], z, name='0', **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('hdf5', type=str, help='input HDF5 file')
    parser.add_argument('zarr', nargs='?', type=str, default=None,
                        help='output path, defaults to random unique suffix')

    # chunk
    parser.add_argument('--chunks', type=int, nargs='*', default=[],
                        help='Chunk shape : list of positive integers (if not provided, will be guessed.)')

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

    args = parser.parse_args()

    if args.zarr is None:
        # get unique new filename
        import uuid
        args.zarr = args.hdf5[:-5] + str(uuid.uuid4()) + '.zarr'

    # chunks: max 5D, all positive int
    args.chunks = tuple(int(c) for c in args.chunks if int(c) > 0)
    if len(args.chunks) > 5:
        raise ValueError('too many values for chunks')
    if len(args.chunks) == 0:
        args.chunks = True

    # compressor : None, default or some evaluated object
    if args.compressor == 'none':
        args.compressor = None
    elif args.compressor != 'default':
        # Zlib, GZip, BZ2, LZMA?, Blosc, Zstd, lz4, ZFPY, and a lot of others...
        import numcodecs
        import codecs
        args.compressor = eval(args.compressor)
    
    # remove warning : copy these argument and don't pass them to .create_dataset
    h_file = args.hdf5
    z_file = args.zarr

    del args.hdf5
    del args.zarr
    
    convert(h_file, z_file, **args.__dict__)
