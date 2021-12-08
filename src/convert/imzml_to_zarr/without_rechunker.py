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


import abc
import timeit
from pathlib import Path
from typing import List, Tuple

import numpy as np
import zarr
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser

from ...utils import temp_stores

_DISK_COPY_THRESHOLD = 8 * 10 ** 9

SHAPE = Tuple[int, int, int, int]

VERSION = 'experiment'


def copy_array(source: zarr.Array, destination: zarr.Array, limit: int = _DISK_COPY_THRESHOLD) -> None:
    "copy a array (ragged arrays non supported)"
    array_size = source.nbytes  # Zarr can be trusted here
    if array_size <= limit:
        # load all data in memory then write at once
        #   - usually faster
        start = timeit.default_timer()
        destination[:] = source[:]
        end = timeit.default_timer()
        print(f'\trechunking (in mem): {end-start: 5.2f} ({array_size=})')
    else:
        # chunk by chunk loading
        #   - smaller memory footprint
        start = timeit.default_timer()
        destination[:] = source
        end = timeit.default_timer()
        print(f'\trechunking (zarr)  : {end-start: 5.2f} ({array_size=})')


class _BaseImzMLConvertor(abc.ABC):
    "base class hiding the continuous VS processed difference behind polymorphism"

    def __init__(self, root: zarr.Group, name: str, parser: PyImzMLParser,
                 memory_limit: int = _DISK_COPY_THRESHOLD) -> None:
        super().__init__()

        self.root = root
        self.name = name
        self.parser = parser
        self.memory_limit = memory_limit

    @abc.abstractmethod
    def get_labels(self) -> List[str]:
        "return the list of labels associated with the image"

    def add_base_metadata(self) -> None:
        """add some OME-Zarr compliant metadata to the root group:
        - multiscales
        - labels

        as well as custom PIMS - MSI metadata in 'pims-msi'
        """

        # multiscales metadata
        self.root.attrs['multiscales'] = [{
            'version': '0.3',
            'name': self.name,
            # store intensities in dataset 0
            'datasets': [{'path': '0'}, ],
            # NOTE axes attribute may change significantly in 0.4.0
            'axes': ['c', 'z', 'y', 'x'],
            'type': 'none',  # no downscaling (at the moment)
        }]

        self.root.attrs['pims-msi'] = {
            'version': VERSION,
            'source': self.parser.filename,
            # image resolution ?
            'uuid': self.parser.metadata.file_description.cv_params[0][2],
            # find out if imzML come from a conversion, include it if so ?
        }

        # label group
        self.root.create_group('labels').attrs['labels'] = self.get_labels()

    @abc.abstractmethod
    def create_zarr_arrays(self):
        """generate empty arrays inside the root group
        """

    @abc.abstractmethod
    def read_binary_data(self) -> None:
        """fill in the arrays defined with the ibd file from the source

        NOTE: missing coordinates will not write to the array, make sure the
        current value for the array is suitable.
        """

    def run(self) -> None:
        "main method"
        self.add_base_metadata()
        self.create_zarr_arrays()
        self.read_binary_data()


class _ContinuousImzMLConvertor(_BaseImzMLConvertor):
    def get_labels(self) -> List[str]:
        return ['mzs/0']

    def get_intensity_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the intensity array"
        return (
            self.parser.mzLengths[0],                        # c = m/Z
            1,                                               # z = 1
            self.parser.imzmldict['max count of pixels y'],  # y
            self.parser.imzmldict['max count of pixels x'],  # x
        )

    def get_mz_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the mzs array"
        return (
            self.parser.mzLengths[0],  # c = m/Z
            1,                         # z
            1,                         # y
            1,                         # x
        )

    def create_zarr_arrays(self):
        """generate empty arrays inside the root group
        """

        # array for the intensity values (main image)
        intensities = self.root.zeros(
            '0',
            shape=self.get_intensity_shape(),
            dtype=self.parser.intensityPrecision,
            # default chunks & compressor
        )

        # xarray zarr encoding
        intensities.attrs['_ARRAY_DIMENSIONS'] = _get_xarray_axes(self.root)

        # array for the m/Z (as a label)
        self.root.zeros(
            'labels/mzs/0',
            shape=self.get_mz_shape(),
            dtype=self.parser.mzPrecision,
            # default chunks
            compressor=None,
        )

        # # NOTE: for now, z axis is supposed to be a Zero for all values
        # # array for z value (as a label)
        # z_values = self.root.zeros(
        #     'labels/z/0',
        #     shape=self.get_z_shape(),
        #     dtype=float,
        #     compressor=None,
        # )

    def read_binary_data(self) -> None:
        intensities = self.root[0]
        mzs = self.root.labels.mzs[0]
        with temp_stores.temp_store() as fast_store:
            # create an array for the temporary intensities
            fast_intensities = zarr.group(fast_store).zeros(
                '0',
                shape=intensities.shape,
                dtype=intensities.dtype,
                chunks=(-1, 1, 1, 1),  # similar to the .ibd structure
                compressor=None,
            )

            start = timeit.default_timer()

            # fill m/Z into the destination group
            self.parser.m.seek(self.parser.mzOffsets[0])
            mzs[:, 0, 0, 0] = np.fromfile(self.parser.m, count=self.parser.mzLengths[0],
                                          dtype=self.parser.mzPrecision)

            # fill intensities into the fast group
            for idx, (x, y, _) in enumerate(self.parser.coordinates):
                self.parser.m.seek(self.parser.intensityOffsets[idx])
                fast_intensities[:, 0, y-1, x-1] = np.fromfile(
                    self.parser.m, count=self.parser.intensityLengths[idx],
                    dtype=self.parser.intensityPrecision)

            end = timeit.default_timer()
            print(
                f'\treading done in {end-start: 5.2f} (processed, len={len(self.parser.coordinates)})')

            # re-chunk
            copy_array(fast_intensities, intensities, limit=self.memory_limit)


class _ProcessedImzMLConvertor(_BaseImzMLConvertor):
    def get_labels(self) -> List[str]:
        return ['mzs/0', 'lengths/0']

    def get_intensity_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the intensity array"
        return (
            max(self.parser.mzLengths),                      # c = m/Z
            1,                                               # z = 1
            self.parser.imzmldict['max count of pixels y'],  # y
            self.parser.imzmldict['max count of pixels x'],  # x
        )

    def get_mz_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the mzs array"
        return self.get_intensity_shape()

    def get_lengths_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the lengths array"
        return (
            1,                                               # c = m/Z
            1,                                               # z = 1
            self.parser.imzmldict['max count of pixels y'],  # y
            self.parser.imzmldict['max count of pixels x'],  # x
        )

    def create_zarr_arrays(self):
        """generate empty arrays inside the root group
        """

        # array for the intensity values (main image)
        intensities = self.root.zeros(
            '0',
            shape=self.get_intensity_shape(),
            dtype=self.parser.intensityPrecision,
            # default chunks & compressor
        )

        # xarray zarr encoding
        intensities.attrs['_ARRAY_DIMENSIONS'] = _get_xarray_axes(self.root)

        # array for the m/Z (as a label)
        self.root.zeros(
            'labels/mzs/0',
            shape=self.get_mz_shape(),
            dtype=self.parser.mzPrecision,
            # default chunks & compressor
        )

        # # NOTE: for now, z axis is supposed to be a Zero for all values
        # # array for z value (as a label)
        # z_values = self.root.zeros(
        #     'labels/z/0',
        #     shape=self.get_z_shape(),
        #     dtype=float,
        #     compressor=None,
        # )

        # array for the lengths (as a label)
        self.root.zeros(
            'labels/lengths/0',
            shape=self.get_lengths_shape(),
            dtype=np.uint32,
            # default chunks
            compressor=None,
        )

    def read_binary_data(self) -> None:
        intensities = self.root[0]
        mzs = self.root.labels.mzs[0]
        lengths = self.root.labels.lengths[0]

        with temp_stores.temp_store() as fast_store:
            fast_group = zarr.group(fast_store)

            # create arrays for the temporary intensities & masses
            fast_intensities = fast_group.zeros(
                '0',
                shape=intensities.shape,
                dtype=intensities.dtype,
                chunks=(-1, 1, 1, 1),  # similar to the .ibd structure
                compressor=None,
            )
            fast_mzs = fast_group.zeros(
                'mzs',
                shape=mzs.shape,
                dtype=mzs.dtype,
                chunks=(-1, 1, 1, 1),
                compressor=None,
            )

            start = timeit.default_timer()

            # read the data into the fast arrays
            for idx, (x, y, _) in enumerate(self.parser.coordinates):
                length = self.parser.mzLengths[idx]
                lengths[0, 0, y-1, x-1] = length
                spectra = self.parser.getspectrum(idx)
                fast_mzs[:length, 0, y-1, x-1] = spectra[0]
                fast_intensities[:length, 0, y-1, x-1] = spectra[1]

            end = timeit.default_timer()
            print(f'\treading done in {end-start: 5.2f} (processed, '
                  f'len={len(self.parser.coordinates)})')

            # re-chunk
            copy_array(fast_intensities, intensities, limit=self.memory_limit)
            copy_array(fast_mzs, mzs, limit=self.memory_limit)


def _get_xarray_axes(root: zarr.Group) -> List[str]:
    "return a copy of the 'axes' multiscales metadata, used for XArray"
    return root.attrs['multiscales'][0]['axes']


def _convert_processed(
    parser: PyImzMLParser,
    zarr_group: zarr.Group,
    max_mem: int = _DISK_COPY_THRESHOLD,
    **kwargs
) -> None:
    """- parser should have all the information for the imzML, already parsed
    - zarr_group represent a group that can be used to create file at the correct
    location
    - shape, dtype, etc should not be in kwargs
    """

    _ProcessedImzMLConvertor(zarr_group, Path(
        parser.filename).stem, parser, memory_limit=max_mem).run()


def _convert_continuous(
    parser: PyImzMLParser,
    zarr_group: zarr.Group,
    max_mem: int = _DISK_COPY_THRESHOLD,
    **kwargs
) -> None:
    """- parser should have all the information for the imzML, already parsed
    - zarr_group represent a group that can be used to create file at the correct
    location
    - shape, dtype, etc should not be in kwargs
    """

    _ContinuousImzMLConvertor(zarr_group, Path(
        parser.filename).stem, parser, memory_limit=max_mem).run()
