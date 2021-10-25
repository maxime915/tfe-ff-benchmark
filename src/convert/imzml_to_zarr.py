"imzml_to_zarr: converts imzML images to Zarr files"

import argparse
import typing

import numpy
import pyimzml.ImzMLParser as imzML
import zarr


def converter(imzml_path: str, zarr_path: str, **kwargs) -> typing.Callable[[], None]:
    "returns a function that does the convertion from imzML to Zarr"

    include_spectra_metadata = kwargs.pop('include-spectra-metadata', None)
    overwrite = kwargs.pop('overwrite', True)

    if 'shape' in kwargs:  # TODO add warning
        kwargs.pop('shape', None)
    if 'dtype' in kwargs:  # TODO add warning
        kwargs.pop('dtype', None)

    # TODO evaluate performance... are there a lot of active threads ?
    # why aren't any of them hitting close to 100% activity ? [60-80]% CPU usage, total

    def converter_fun() -> None:

        # the ImzMLParser object automatically parses the whole .imzML file at once
        # this creates a huge slow-down when most of it could potentially be parsed
        # online & in parallel to allow parallel conversion.
        parser = imzML.ImzMLParser(
            imzml_path, 'lxml', include_spectra_metadata=include_spectra_metadata)

        # TODO processed mode to be implemented, will there ever be another one ?

        # check for continuous binary mode
        if not parser.metadata.file_description.param_by_name.get('continuous', False):
            raise ValueError(
                'unsupported imzML file: mode should be continuous')

        # TODO are there any other type supported by the standard

        # check for dtype
        if not parser.metadata.referenceable_param_groups['intensities'].\
                param_by_name.get('32-bit float', False):
            raise ValueError(
                'unsupported imzML file: intensities should have float32 dtype.')
        if not parser.metadata.referenceable_param_groups['mzArray'].\
                param_by_name.get('64-bit float', False):
            raise ValueError(
                'unsupported imzML file: mzArray should have float64 dtype.')

        # TODO imzML seems to support full 3D image... is it the case ?
        # obtained "3D" dataset just have long m/z bands,
        # z coordinate is still zero for all sample

        # check for 2d planes
        if parser.imzmldict.get('max count of pixels z', 1) != 1:
            raise ValueError(
                'unsupported imzML file: should have 1 z dimension')

        # get amount of m/z values per pixel
        shape = (parser.imzmldict['max count of pixels x'],
                 parser.imzmldict['max count of pixels y'],
                 parser.mzLengths[0])

        store = zarr.DirectoryStore(zarr_path)
        zarr_file = zarr.group(store=store, overwrite=overwrite)

        # (partial) OME-Zarr attributes
        zarr_file.attrs['multiscales'] = {
            'version': '0.3',
            'name': zarr_path,
            'datasets': [{'path': '0'}, {'path': 'mzs'}],
            'axes': ['x', 'y', 'c'],
            'type': 'conversion from imzML file',
            'metadata': {'original': imzml_path},  # TODO include UUID
        }
        # TODO Omero metadata?

        array3d = zarr_file.create(
            '0', shape=shape, dtype=numpy.float32, **kwargs)
        mzs = zarr_file.create(
            'mzs', shape=shape[2:], chunks=True, dtype=numpy.float64)

        # copy a single m/z sequence : this file is continuous so its the same for all (x, y)
        mzs[:] = parser.coordinates[0][0]

        for idx, (x, y, _) in enumerate(parser.coordinates):
            _, intensities = parser.getspectrum(idx)
            array3d[x-1, y-1, :] = intensities

    return converter_fun


def convert(imzml_path: str, zarr_path: str, **kwargs) -> None:
    "convert an imzML image to a Zarr"
    converter(imzml_path, zarr_path, **kwargs)()


if __name__ == "__main__":
    # TODO more parsing options ?
    argparser = argparse.ArgumentParser()
    argparser.add_argument('imzml', type=str, nargs=1)
    argparser.add_argument('zarr', type=str, nargs=1)
    args = argparser.parse_args()
    convert(args.imzml, args.zarr)
