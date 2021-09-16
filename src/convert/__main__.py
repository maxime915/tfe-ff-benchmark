"utils to convert a set of HDF5 file to Zarr format"

import sys

from . import hdf5_to_zarr

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit('at least one argument required')
    for file in sys.argv[1:]:

        zarr_file = hdf5_to_zarr(file)
        print(f'{file} converted')
