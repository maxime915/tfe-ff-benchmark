"imzml_to_zarr: converts imzML images to Zarr files"

import argparse
import sys
import timeit
import uuid
import warnings
from typing import Callable

import numcodecs
import numpy as np
import zarr
from pyimzml import ImzMLParser


def _convert_processed(
    parser: ImzMLParser.ImzMLParser,
    zarr_group: zarr.Group,
    **kwargs
) -> None:
    """- parser should have all the information for the imzML, already parsed
    - zarr_group represent a group that can be used to create file at the correct
    location
    - shape, dtype, etc should not be in kwargs
    """

    shape = (parser.imzmldict['max count of pixels x'],
             parser.imzmldict['max count of pixels y'])

    temp_store = zarr.TempStore()
    temp_group = zarr.group(temp_store, overwrite=False)

    # use a copy of the parameters
    temp_param = kwargs.copy()
    temp_param['chunks'] = (1, 1)
    temp_param['shape'] = shape
    temp_param['compressor'] = None

    # create temporary ragged arrays -> fast write to chunks
    temp_int = temp_group.empty('intensities', object_codec=numcodecs.VLenArray(
        parser.intensityPrecision), dtype=object, **temp_param)
    temp_mzs = temp_group.empty('mzs', object_codec=numcodecs.VLenArray(
        parser.mzPrecision), dtype=object, **temp_param)

    for idx, (x, y, _) in enumerate(parser.coordinates):
        temp_mzs[x-1, y-1], temp_int[x-1, y-1] = parser.getspectrum(idx)

    # make sure the caller knows that 3rd axis has full chunk (un-supported)
    if 'chunks' in kwargs and isinstance(kwargs['chunks'], (tuple, list)) and len(kwargs['chunks']) > 2:
        if kwargs['chunks'][2] != -1:
            warnings.warn(f'unsupported chunk {kwargs["chunks"]} for processed imzml, third axis will be full')
        kwargs['chunks'] = kwargs['chunks'][:2]

    # create ragged arrays for final storage
    intensities = zarr_group.empty('intensities', object_codec=numcodecs.VLenArray(
        parser.intensityPrecision), shape=shape, dtype=object, **kwargs)
    mzs = zarr_group.empty('mzs', object_codec=numcodecs.VLenArray(
        parser.mzPrecision), shape=shape, dtype=object, **kwargs)

    # re-chunk
    intensities[:] = temp_int[:]
    mzs[:] = temp_mzs

    # remove temporary store (the automatic removal is a shutdown only)
    temp_store.rmdir()


def _convert_continuous(
    parser: ImzMLParser.ImzMLParser,
    zarr_group: zarr.Group,
    **kwargs
) -> None:
    """- parser should have all the information for the imzML, already parsed
    - zarr_group represent a group that can be used to create file at the correct
    location
    - shape, dtype, etc should not be in kwargs
    """

    shape = (parser.imzmldict['max count of pixels x'],
             parser.imzmldict['max count of pixels y'],
             parser.mzLengths[0])

    temp_store = zarr.TempStore()
    temp_group = zarr.group(temp_store, overwrite=False)
    temp_int = temp_group.empty('intensities', shape=shape, dtype=parser.intensityPrecision,
                                chunks=(1, 1, -1), compressor=None)

    # create array to be filled -> use default options
    mzs = zarr_group.create('mzs', shape=shape[2:], dtype=parser.mzPrecision)

    # fill m/z
    parser.m.seek(parser.mzOffsets[0])
    mzs[:] = np.fromfile(parser.m, count=parser.mzLengths[0],
                         dtype=parser.mzPrecision)

    for idx, (x, y, _) in enumerate(parser.coordinates):

        # manual call to seek instead of offset: repeatidly calling fromfile
        # with the same offset drifts, this is not the expected behavior so
        # the file is seek'ed manually before reading

        parser.m.seek(parser.intensityOffsets[idx])
        temp_int[x-1, y-1, :] = np.fromfile(
            parser.m, count=parser.intensityLengths[idx], dtype=parser.intensityPrecision)

    # create array for final storage
    intensities = zarr_group.empty('intensities', shape=shape,
                                   dtype=parser.intensityPrecision, **kwargs)

    # re-chunk
    intensities[:] = temp_int[:]

    # cleanup
    temp_store.rmdir()


def converter(imzml_path: str, zarr_path: str, **kwargs) -> Callable[[], None]:
    "returns a function that does the convertion from imzML to Zarr"

    include_spectra_metadata = kwargs.pop('include-spectra-metadata', None)
    overwrite = kwargs.pop('overwrite', True)

    for key in ['shape', 'dtype', 'max-shape', 'max_shape']:
        if key in kwargs:
            kwargs.pop(key, None)
            warnings.warn(f'"{key}" argument ignored in imzML -> Zarr conversion')

    def converter_fun() -> None:

        # the ImzMLParser object automatically parses the whole .imzML file at once
        with warnings.catch_warnings(record=True) as _:
            parser = ImzMLParser.ImzMLParser(imzml_path, 'lxml',
                                             include_spectra_metadata=include_spectra_metadata)

        # imzML seems to support full 3D image... is it the case ?
        # obtained "3D" dataset just have long m/z bands,
        # z coordinate is still zero for all sample

        # check for 2d planes
        if parser.imzmldict.get('max count of pixels z', 1) != 1:
            raise ValueError('unsupported imzML file: should have 1 z dimension')

        # check for binary mode
        is_continuous = 'continuous' in parser.metadata.file_description.param_by_name
        is_processed = 'processed' in parser.metadata.file_description.param_by_name

        if is_processed == is_continuous:
            raise ValueError('unsupported imzML file: mode should be continuous or processed')

        store = zarr.DirectoryStore(zarr_path)
        zarr_file = zarr.group(store=store, overwrite=overwrite)

        # (partial) OME-Zarr attributes
        zarr_file.attrs['multiscales'] = {
            'version': '0.3',
            'name': zarr_path,
            'datasets': [{'path': 'intensities'}, {'path': 'mzs'}],
            'axes': ['x', 'y', 'c'],
            'type': 'conversion from imzML file',
            'metadata': {'original': imzml_path},  # include UUID ?
        }
        # Omero metadata?

        if is_continuous:
            _convert_continuous(parser, zarr_file, **kwargs)
        else:
            _convert_processed(parser, zarr_file, **kwargs)

        # close underlining idb file
        parser.m.close()

    return converter_fun


def convert(imzml_path: str, zarr_path: str, **kwargs) -> None:
    "convert an imzML image to a Zarr"
    converter(imzml_path, zarr_path, **kwargs)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('imzml', type=str, help='input imzML file')
    parser.add_argument('zarr', nargs='?', type=str, default=None,
                        help='output path, defaults to random unique suffix')

    # chunk
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument('--auto-chunk', dest='auto_chunk', action='store_true',
                             help='automatic chunking for array storage (default option)')
    chunk_group.add_argument('--no-chunk', dest='auto_chunk', action='store_false',
                             help='disable chunking for array storage')
    chunk_group.add_argument('--chunks', type=int, nargs='+', default=[],
                             help='Chunk shape : list of positive integers (-1 means equal to array shape)')
    parser.set_defaults(auto_chunk=True)

    # rand name
    rand_name = parser.add_mutually_exclusive_group()
    rand_name.add_argument('--add-rand-suffix', dest='rand_name', action='store_true',
                           help='if zarr is not specified, guess the zarr file by adding a random UUID4 suffix and changing the extension')
    rand_name.add_argument('--no-rand-suffix', dest='rand_name', action='store_false',
                           help='if zarr is not specified, guess the zarr file by changing the extension (default)')
    parser.set_defaults(rand_name=False)

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
        if args.rand_name:
            args.zarr = args.imzml[:-4] + str(uuid.uuid4()) + '.zarr'
        else:
            args.zarr = args.imzml[:-4] + 'zarr'

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
        import codecs

        try:
            args.compressor = eval(args.compressor)
        except Exception as e:
            print(f'unable to evaluate compressor\n\n{e}')

    if args.benchmark:
        results = timeit.Timer(converter(args.imzml, args.zarr, chunks=chunks, compressor=args.compressor,
                               cache_metadata=args.cache_metadata, order=args.order)).repeat(3, number=10)
        results = np.array(results) / 10
        res = "[" + ", ".join([f'{v:5.3e}' for v in results]) + "]"
        print((f'results: {res}: {results.min():5.3e} s to {results.max():5.3e} s, '
               f'{results.mean():5.3e} s ± {results.std(ddof=1):5.3e} s'))
    else:
        time = timeit.Timer(converter(args.imzml, args.zarr, chunks=chunks, compressor=args.compressor,
                                      cache_metadata=args.cache_metadata, order=args.order)).timeit(1)
        print(f'converted {args.imzml} to {args.zarr} in {time:.4}s')
