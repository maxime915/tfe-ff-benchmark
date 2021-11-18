"""without rechunker: converting imzML + ibd files to Zarr groups without rechunker

2 stage conversion process [genereal idea]:

    - create a Zarr array with no compressor, and chunks similar to the spectrums
    - parse the whole .imzML file
    - read each spectrum one by one, write it to the previous Zarr array
    - create a Zarr array with the required parameters (compressor, order, chunks)
    - copy the first array to the second one


Using simple copy :
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

import timeit

import numcodecs
import numpy as np
import zarr
from pyimzml import ImzMLParser
from zarr.util import guess_chunks


_DISK_COPY_THRESHOLD = 4 * 10 ** 9


def _convert_processed(
    parser: ImzMLParser.ImzMLParser,
    zarr_group: zarr.Group,
    max_mem: int = _DISK_COPY_THRESHOLD,
    **kwargs
) -> None:
    """- parser should have all the information for the imzML, already parsed
    - zarr_group represent a group that can be used to create file at the correct
    location
    - shape, dtype, etc should not be in kwargs
    """

    shape = (parser.imzmldict['max count of pixels x'],
             parser.imzmldict['max count of pixels y'])

   # create a temporary store for the first stage of the conversion
    fast_store = zarr.TempStore()
    fast_group = zarr.group(fast_store, overwrite=False)
    fast_intensities = fast_group.empty(
        'intensities',
        shape=shape,
        dtype=object,
        object_codec=numcodecs.VLenArray(parser.intensityPrecision),
        chunks=(1, 1),
        compressor=None
    )
    fast_mzs = fast_group.empty(
        'mzs',
        shape=shape,
        dtype=object,
        object_codec=numcodecs.VLenArray(parser.mzPrecision),
        chunks=(1, 1),
        compressor=None
    )

    read_start = timeit.default_timer()

    # fill m/Z & intensities into the fast group
    sumlen = 0
    for idx, (x, y, _) in enumerate(parser.coordinates):
        intensities, mzs = parser.getspectrum(idx)
        sumlen += len(mzs)

        fast_mzs[x-1, y-1] = mzs
        fast_intensities[x-1, y-1] = intensities

    read_end = timeit.default_timer()

    # chunks for intensity
    chunks = kwargs.pop('chunks', True)
    meansize = sumlen * \
        np.dtype(parser.intensityPrecision).itemsize // len(parser.coordinates)
    if chunks == True:
        chunks = guess_chunks(shape, meansize)

    # re-chunk
    dest_intensities = zarr_group.empty(
        'intensities',
        shape=shape,
        chunks=chunks,
        dtype=object,
        object_codec=numcodecs.VLenArray(parser.intensityPrecision),
        **kwargs
    )
    dest_mzs = zarr_group.empty(
        'mzs',
        shape=shape,
        dtype=object,
        chunks=chunks,
        object_codec=numcodecs.VLenArray(parser.mzPrecision),
        **kwargs
    )

    # zarr cannot be trusted for object array: need to recompute the max size
    array_size = max(
        np.dtype(parser.intensityPrecision).itemsize,
        np.dtype(parser.mzPrecision).itemsize
    ) * sumlen
    if array_size <= max_mem:
        # load all in memory then write at once
        dest_intensities[:] = fast_intensities[:]
        dest_mzs[:] = fast_mzs[:]
        branch = "in memory"
    else:
        # let Zarr do the loading based on chunks
        dest_intensities[:] = fast_intensities
        dest_mzs[:] = fast_mzs
        branch = "used zarr"

    end_rechunk = timeit.default_timer()

    print(f'done {branch=}')
    print(f'reading     : {read_end-read_start: 5.2f}s\trechunking  : {end_rechunk-read_end: 5.2f}s')

    # remove temporary stores
    fast_store.rmdir()


def _convert_continuous(
    parser: ImzMLParser.ImzMLParser,
    zarr_group: zarr.Group,
    max_mem: int = _DISK_COPY_THRESHOLD,
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

    # create m/Z array
    dest_mzs = zarr_group.empty(
        'mzs',
        shape=shape[-1:],
        dtype=parser.mzPrecision,
        # default chunks is fine
        compressor=None,  # small array, little gain to compression
    )

    # chunks for intensity
    chunks = kwargs.pop('chunks', True)
    if chunks == True:
        chunks = guess_chunks(shape, np.dtype(
            parser.intensityPrecision).itemsize)

    # create a temporary store for the first stage of the conversion
    fast_store = zarr.TempStore()
    fast_group = zarr.group(fast_store, overwrite=False)
    fast_intensities = fast_group.empty('intensities', shape=shape, dtype=parser.intensityPrecision,
                                        chunks=(1, 1, chunks[-1]), compressor=None)

    read_start = timeit.default_timer()

    # fill m/Z into the destination group
    parser.m.seek(parser.mzOffsets[0])
    dest_mzs[:] = np.fromfile(parser.m, count=parser.mzLengths[0],
                              dtype=parser.mzPrecision)

    # fill intensities into the fast group
    for idx, (x, y, _) in enumerate(parser.coordinates):
        parser.m.seek(parser.intensityOffsets[idx])
        fast_intensities[x-1, y-1, :] = np.fromfile(parser.m, count=parser.intensityLengths[idx],
                                                    dtype=parser.intensityPrecision)

    read_end = timeit.default_timer()

    # re-chunk
    dest_intensities = zarr_group.empty(
        'intensities',
        shape=shape,
        dtype=parser.intensityPrecision,
        chunks=chunks,
        **kwargs
    )

    array_size = fast_intensities.nbytes  # Zarr can be trusted here
    if array_size <= max_mem:
        # load all in memory then write at once
        dest_intensities[:] = fast_intensities[:]
    else:
        # let Zarr do the loading based on chunks
        dest_intensities[:] = fast_intensities

    end_rechunk = timeit.default_timer()

    print('done')
    print(f'reading     : {read_end-read_start: 5.2f}s\trechunking  : {end_rechunk-read_end: 5.2f}s')

   # remove temporary store
    fast_store.rmdir()
