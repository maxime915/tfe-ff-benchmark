"""imzml_to_zarr: converting imzML + ibd files to Zarr groups

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

Using rechunker (see with_rechunker.py):
    - temporary Zarr has chunks (1, 1, -1) for continuous and (1, 1) for processed
    - rechunking is done via rechunker [bad support for VLA]
    - allow to limit the total used memory
    - memory support for VLA is not working (may use too much memory and fail)

Using simple copy (see without_rechunker.py):
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

import warnings
from typing import Callable

import zarr
from pyimzml import ImzMLParser

from .with_rechunker import _convert_continuous as convert_continuous_rc
from .with_rechunker import _convert_processed as convert_processed_rc
from .without_rechunker import _convert_continuous as convert_continuous_nr
from .without_rechunker import _convert_processed as convert_processed_nr


def converter(imzml_path: str, zarr_path: str, max_mem: int = -1, use_rechunker=False, **kwargs) -> Callable[[], None]:
    "returns a function that does the convertion from imzML to Zarr"

    include_spectra_metadata = kwargs.pop('include-spectra-metadata', None)
    overwrite = kwargs.pop('overwrite', True)

    for key in ['shape', 'dtype', 'max-shape', 'max_shape', 'object_codec']:
        if key in kwargs:
            warnings.warn(f'ignoring kwargs["{key}"] = {kwargs[key]}')
            kwargs.pop(key)

    _convert_continuous = convert_continuous_rc if use_rechunker else convert_continuous_nr
    _convert_processed = convert_processed_rc if use_rechunker else convert_processed_nr

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
            raise ValueError('unsupported imzML: z shape must be 1')

        # check for binary mode
        is_continuous = 'continuous' in parser.metadata.file_description.param_by_name
        is_processed = 'processed' in parser.metadata.file_description.param_by_name

        if is_processed == is_continuous:
            raise ValueError('unsupported imzML: invalid binary mode')

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

        if max_mem > 0:
            kwargs['max_mem'] = max_mem

        if is_continuous:
            _convert_continuous(parser, zarr_file, **kwargs)
        else:
            _convert_processed(parser, zarr_file, **kwargs)

        # close underlining idb file
        parser.m.close()

    return converter_fun


def convert(imzml_path: str, zarr_path: str, use_rechunker=True, **kwargs) -> None:
    "convert an imzML image to a Zarr"
    converter(imzml_path, zarr_path, use_rechunker, **kwargs)()
