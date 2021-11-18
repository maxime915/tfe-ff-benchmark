"""with_rechunker: converting imzML + ibd files to Zarr groups using rechunker

2 stage conversion process [genereal idea]:

    - create a Zarr array with no compressor, and chunks similar to the spectrums
    - parse the whole .imzML file
    - read each spectrum one by one, write it to the previous Zarr array
    - create a Zarr array with the required parameters (compressor, order, chunks)
    - copy the first array to the second one


Using rechunker:
    - temporary Zarr has chunks (1, 1, -1) for continuous and (1, 1) for processed
    - rechunking is done via rechunker [bad support for VLA]
    - allow to limit the total used memory
    - memory support for VLA is not working (may use too much memory and fail)
"""

import timeit
import warnings

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
    max_mem: int = _MAX_MEMORY,
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
            warnings.warn(f'unsupported chunk {kwargs["chunks"]} for processed imzml, third axis will be full')
        kwargs['chunks'] = kwargs['chunks'][:2]

    # guess chunks
    chunks = kwargs.pop('chunks', True)
    meansize = sumlen * \
        np.dtype(parser.intensityPrecision).itemsize // len(parser.coordinates)
    if chunks == True:
        # mzs & intensities have the same shape & chunks to have a file bijection
        chunks = guess_chunks(shape, meansize)

    # reformat chunks for rechunker
    chunks = [c if c > 0 else s for c, s in zip(chunks, shape)]

    temp_store = zarr.TempStore()

    # rechunker assumes that the object dtype is 'O' which has size 8.
    #   In reality, the whole sub-array is much larger than that.
    #   The limit is divided by the mean size of the sub arrays to restore the memory limmit

    mem_limit = (max_mem * np.dtype('O').itemsize) // meansize

    rechunked = rechunker.rechunk(
        source=fast_group,
        target_chunks={
            'intensities': chunks,
            'mzs': chunks,
        },
        max_mem=mem_limit,
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
    print(f'reading     : {read_end-read_start: 5.2f}s\trechunking  : {end_rechunk-read_end: 5.2f}s')

    # remove temporary stores
    temp_store.rmdir()
    fast_store.rmdir()


def _convert_continuous(
    parser: ImzMLParser.ImzMLParser,
    zarr_group: zarr.Group,
    max_mem: int = _MAX_MEMORY,
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
        max_mem=max_mem,
        target_store=zarr_group.store,
        target_options={
            'intensities': kwargs,
        },
        temp_store=temp_store)
    rechunked.execute()

    end_rechunk = timeit.default_timer()

    print('done')
    print(f'reading     : {read_end-read_start: 5.2f}s\trechunking  : {end_rechunk-read_end: 5.2f}s')

   # remove temporary stores
    fast_store.rmdir()
    temp_store.rmdir()
