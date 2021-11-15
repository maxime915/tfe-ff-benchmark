"""imzml_to_zarr: converts imzML images to Zarr files

2 stage conversion process [genereal idea]:

    - create a Zarr array with no compressor, and chunks similar to the spectrums
    - parse the whole .imzML file
    - read each spectrum one by one, write it to the previous Zarr array
    - create a Zarr array with the required parameters (compressor, order, chunks)
    - copy the first array to the second one


Full in-memory transit [fast but prohibitive due to memory usage]:
    - temporary Zarr has chunks (1, 1, -1) for continuous and (1, 1) for processed
    - to rechunk: load the whole array in memory, then write it back to the destination
    - possibly high memory usage
    - much faster than doing the one stage conversion

Using rechunker [current]:
    - temporary Zarr has chunks (1, 1, -1) for continuous and (1, 1) for processed
    - rechunking is done via rechunker [bad support for VLA]
    - allow to limit the total used memory
    - memory support for VLA is not working (may use too much memory and fail)

Using zarr.copy [todo]:
    - create an empty destination Array with the desired parameters
    - create a temporary Zarr with chunks (1, 1, X) [cont.] or (1, 1) [proc.] where
        X is the value from the destination Array's chunks
    - read the spectrums to the temporary Array
    IF HIGH SIZE ARRAY:
        - use zarr.copy function to make the copy
        - one-to-many mapping between chunks (each write requires multiple read, but
            each temporary chunk is only read once)
        - no memory limitation but the output chunks should fit in memory and 
            this approach shouldn't use more than that
    ELSE:
        - load array in memory
        - write it to the destination directly
"""

import argparse
import sys
import timeit
import uuid
import warnings
from typing import Callable

import numcodecs
import numpy as np
import rechunker
import zarr
from pyimzml import ImzMLParser
from zarr.util import guess_chunks

# 2GiB - 1 (to satisfy Blosc compressor limit)
_MAX_MEMORY = 2147483647


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

    fast_store = zarr.TempStore()
    fast_group = zarr.group(fast_store, overwrite=False)

    # use a copy of the parameters
    fast_params = kwargs.copy()
    fast_params['chunks'] = (1, 1)
    fast_params['shape'] = shape
    fast_params['compressor'] = None

    # create temporary ragged arrays -> fast write to chunks
    fast_int = fast_group.empty('intensities', object_codec=numcodecs.VLenArray(
        parser.intensityPrecision), dtype=object, **fast_params)
    fast_mzs = fast_group.empty('mzs', object_codec=numcodecs.VLenArray(
        parser.mzPrecision), dtype=object, **fast_params)

    read_start = timeit.default_timer()

    # read data into fast arrays
    sumlen = 0
    for idx, (x, y, _) in enumerate(parser.coordinates):
        intensities, mzs = parser.getspectrum(idx)
        sumlen += len(mzs)

        fast_mzs[x-1, y-1] = mzs
        fast_int[x-1, y-1] = intensities

    read_end = timeit.default_timer()

    # only 2D chunks (or chunks with -1 as their last args)
    if 'chunks' in kwargs and isinstance(kwargs['chunks'], (tuple, list)) and len(kwargs['chunks']) > 2:
        if kwargs['chunks'][2] != -1:
            warnings.warn(
                f'unsupported chunk {kwargs["chunks"]} for processed imzml, third axis will be full')
        kwargs['chunks'] = kwargs['chunks'][:2]

    # guess chunks
    chunks = kwargs.pop('chunks', True)
    if chunks == True:
        # mzs & intensities have the same shape & chunks to have a file bijection
        meansize = sumlen * \
            np.dtype(parser.intensityPrecision).itemsize // len(parser.coordinates)
        chunks = guess_chunks(shape, meansize)

    # reformat chunks for rechunker
    chunks = [c if c > 0 else s for c, s in zip(chunks, shape)]

    temp_store = zarr.TempStore()

    # rechunker assumes that the object dtype is 'O' which has size 8.
    #   In reality, the whole sub-array is much larger than that.
    #   The limit is divided by the mean size of the sub arrays to restore the memory limmit

    max_mem = (_MAX_MEMORY * np.dtype('0').itemsize) // meansize

    rechunked = rechunker.rechunk(
        source=fast_group,
        target_chunks={
            'intensities': chunks,
            'mzs': chunks,
        },
        max_mem=max_mem,
        target_store=zarr_group.store,
        target_options={
            'intensities': {
                **kwargs,
                'object_codec': numcodecs.VLenArray(parser.intensityPrecision)
            },
            'mzs': {
                **kwargs,
                'object_codec': numcodecs.VLenArray(parser.mzPrecision)
            },
        },
        temp_store=temp_store)
    rechunked.execute()

    end_rechunk = timeit.default_timer()

    print('done')
    print(f'reading     : {read_end-read_start: 5.2f}s')
    print(f'rechunking  : {end_rechunk-read_end: 5.2f}s')

    # remove temporary stores
    temp_store.rmdir()
    fast_store.rmdir()


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

    # create a temporary store for the first stage of the conversion
    fast_store = zarr.TempStore()
    fast_group = zarr.group(fast_store, overwrite=False)
    fast_intensities = fast_group.empty('intensities', shape=shape, dtype=parser.intensityPrecision,
                                        chunks=(1, 1, -1), compressor=None)
    # store m/Z in the final group
    mzs = zarr_group.empty(
        'mzs',
        shape=shape[-1:],
        dtype=parser.mzPrecision,
        # default chunks is fine
        compressor=None,  # small array, little gain to compression
    )

    read_start = timeit.default_timer()

    parser.m.seek(parser.mzOffsets[0])
    mzs[:] = np.fromfile(parser.m, count=parser.mzLengths[0],
                         dtype=parser.mzPrecision)

    # fill intensities into the fast store
    for idx, (x, y, _) in enumerate(parser.coordinates):
        parser.m.seek(parser.intensityOffsets[idx])
        fast_intensities[x-1, y-1, :] = np.fromfile(parser.m, count=parser.intensityLengths[idx],
                                                    dtype=parser.intensityPrecision)

    read_end = timeit.default_timer()

    # guess chunks
    chunks = kwargs.pop('chunks', True)
    if chunks == True:
        chunks = guess_chunks(shape, np.dtype(
            parser.intensityPrecision).itemsize)

    # reformat chunks for rechunker
    chunks = [c if c > 0 else s for c, s in zip(chunks, shape)]

    # temporary store for rechunker
    temp_store = zarr.TempStore()

    rechunked = rechunker.rechunk(
        source=fast_group,
        target_chunks={
            'intensities': chunks,
        },
        max_mem=_MAX_MEMORY,
        target_store=zarr_group.store,
        target_options={
            'intensities': kwargs,
        },
        temp_store=temp_store)
    rechunked.execute()

    end_rechunk = timeit.default_timer()

    print('done')
    print(f'reading     : {read_end-read_start: 5.2f}s')
    print(f'rechunking  : {end_rechunk-read_end: 5.2f}s')

   # remove temporary stores
    fast_store.rmdir()
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
            args.zarr = args.imzml[:-5] + str(uuid.uuid4()) + '.zarr'
        else:
            args.zarr = args.imzml[:-5] + 'zarr'

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
