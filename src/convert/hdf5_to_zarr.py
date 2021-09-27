"hdf5_to_zarr: copy a HDF5 group to a Zarr group using the same structure"

import os
from typing import Union

import h5py
import zarr


# pylint: disable=too-few-public-methods
class CopyVisitor:
    """visitor object associated to a zarr.Group to copy the content of an HDF5 file.
    This conversion re-create the same file structure"""

    def __init__(self, zarr_file: zarr.Group, **kwargs) -> None:
        self.zarr_file = zarr_file
        self.kwargs = kwargs

    def visit(self, name: str, value: Union[h5py.Group, h5py.Dataset]) -> None:
        "visitor function, receives the dataset/group and its name"

        # only copy datasets
        if not isinstance(value, h5py.Dataset):
            return

        # create the copy
        self.zarr_file.create_dataset(name, data=value, **self.kwargs)


def hdf5_to_zarr(hdf5_path: str, **kwargs) -> str:
    """hdf5_to_zarr converts an HDF5 ND-array to a zarr one.
    kwargs will be passed to zarr.Group.create_dataset, it can
    therefore set compression, chunks, store, etc."""

    # auto-tiling by default
    if 'chunks' not in kwargs:
        kwargs['chunks'] = True

    with h5py.File(hdf5_path) as file:
        filename = os.path.splitext(hdf5_path)[0] + '.zarr'
        zarr_file = zarr.open(filename, mode='w')
        visitor = CopyVisitor(zarr_file=zarr_file, **kwargs).visit
        file.visititems(visitor)
        return filename


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.exit('at least one argument required')

    for __file in sys.argv[1:]:
        __zarr_file = hdf5_to_zarr(__file)
        print(f'{__file} converted to {__zarr_file}')
