'ome_tiff_to_zarr : convert an OME-Tiff to Zarr'

import argparse
import sys

import tifffile
import zarr

def convert(t_file: str, z_file: str, **kwargs) -> None:
    tif = tifffile.TiffFile(t_file)
    z = zarr.open(z_file, mode='w')

    z.create_dataset('0', data=[p.asarray() for p in tif.pages], **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    # file in
    parser.add_argument('--input', type=str, help='input OME-Tiff file', required=True)

    # file out
    parser.add_argument('--out', type=str, help='Zarr file name to store', required=True)

    # chunks
    parser.add_argument('--chunk', nargs='+', help='shape of the chunk (or True for default)')

    # compressor TODO add more options
    parser.add_argument('--compressor', choices=['none', 'default'], default='default', nargs='?')

    args = parser.parse_args()

    kwargs = {}
    if args.chunk is not None:
        for i, chunk in enumerate(args.chunk):
            try:
                args.chunk[i] = int(chunk)
                if args.chunk[i] <= 0:
                    raise ValueError()
            except ValueError:
                sys.exit(f'invalid chunk value: {chunk}')
        
        if len(args.chunk) > 3:
            sys.exit(f'too many value for chunk : {len(args.chunk)}')

        kwargs['chunk'] = args.chunk

    kwargs['compressor'] = None
    
    convert(args.input, args.out, **kwargs)
