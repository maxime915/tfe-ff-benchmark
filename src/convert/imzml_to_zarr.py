"imzml_to_zarr: converts imzML images to Zarr files"

import argparse
from typing import Callable
import warnings
import numcodecs

import numpy as np
from pyimzml import ImzMLParser
import zarr


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
    # more parsing options ?
    argparser = argparse.ArgumentParser()
    argparser.add_argument('imzml', type=str)
    argparser.add_argument('zarr', type=str)
    args = argparser.parse_args()
    convert(args.imzml, args.zarr)
