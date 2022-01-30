"present: script to show a graph of the different access types"

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from .inspect_db import DB

if __name__ != "__main__":
    sys.exit("should not be imported")

parser = argparse.ArgumentParser()
parser.add_argument('command', choices=[
    "band",
    "tic",
    "conversion",
], help="command to run")
parser.add_argument('files', type=str, help='shelf store file')

args = parser.parse_args()

# fuse all files (overriding older)
data = DB(args.files[0]).load_all()

command = args.command

if command == "band":
    access_number = data['parameters']['access_number']
    
    # see what was the default chunk shape
    
    chunk_shapes = [dict(v)['Chunk shape'] for k, v in data['conversion'].items() if 'infos' in k]
    # print(chunk_shapes)
    
    mode = "continuous" if chunk_shapes[0].count(',') == 2 else "processed"
    
    # load imzML as a reference
    imzml = np.array(data['imzml_raw']['band']) / access_number

    # load all Zarr variation
    zarr_dict : dict = data['imzml_zarr']
    zarr_dict = { k: v for k, v in zarr_dict.items() if 'band' in k }
   
# NOTE: for processed file, the default chunks are the same as the full chunks
# don't repeat the default one
# the full chunks is (-1, -1, -1) and not (-1, -1, 1) as indicated (unavailable for processed files)
    if mode == "processed":
        for k in list(zarr_dict.keys()):
            if (-1, -1, 1) == k[0]:
                del zarr_dict[k]
            if True == k[0]:
                bkp = zarr_dict[k]
                del zarr_dict[k]
                zarr_dict[('Full chunks',) + k[1:]] = bkp   
 
    # make a bar plot
    names = ['imzml'] + ['\n'.join(str(i) for i in k[:-1]) for k in zarr_dict.keys()]
    values = [imzml.mean()] + [np.array(v).mean() / access_number for v in zarr_dict.values()]
    
    fig = plt.figure(figsize=[10.4, 4.8])
    axis = fig.add_subplot(111)
    
    axis.bar(names, values, width=0.8)
    axis.set_yscale('log')
    axis.set_title(f'Spectral Access time (single band) - {mode} mode file')
    axis.set_ylabel('Mean time (s)')
    
    fig.tight_layout()
    fig.savefig(f'band_{mode}.png')
    

elif command == "tic":
    access_number = data['parameters']['access_number']
    
    # see what was the default chunk shape
    chunk_shapes = [dict(v)['Chunk shape'] for k, v in data['conversion'].items() if 'infos' in k]
    # print(chunk_shapes)
    mode = "continuous" if chunk_shapes[0].count(',') == 2 else "processed"
    
    for tile_width in (16, 32, 64):    
            
        # load imzML as a reference
        imzml = np.array(data['imzml_raw'][(tile_width, tile_width)]) / access_number

        # load all Zarr variation
        zarr_dict = { k: v for k, v in data['imzml_zarr'].items() if (tile_width, tile_width) in k }


# NOTE: for processed file, the default chunks are the same as the full chunks
# don't repeat the default one
# the full chunks is (-1, -1, -1) and not (-1, -1, 1) as indicated (unavailable for processed files)
        if mode == "processed":
            for k in list(zarr_dict.keys()):
                if (-1, -1, 1) == k[0]:
                    del zarr_dict[k]
                if True == k[0]:
                    bkp = zarr_dict[k]
                    del zarr_dict[k]
                    zarr_dict[('Full chunks',) + k[1:]] = bkp
        
        # make a bar plot
        names = ['imzml'] + ['\n'.join(str(i) for i in k[:-1]) for k in zarr_dict.keys()]
        values = [imzml.mean()] + [np.array(v).mean() / access_number for v in zarr_dict.values()]
        
        fig = plt.figure(figsize=[10.4, 4.8])
        axis = fig.add_subplot(111)
        
        axis.bar(names, values, width=0.8)
        axis.set_yscale('log')
        axis.set_title(f'Total Ion Current Access time (sum over all bands) - {mode} mode file')
        axis.set_ylabel('Mean time (s)')
        
        fig.tight_layout()
        fig.savefig(f'tic_{tile_width}_{mode}.png')

elif command == "conversion":
    access_number = data['parameters']['conversion_number']
    
    # see what was the default chunk shape
    chunk_shapes = [dict(v)['Chunk shape'] for k, v in data['conversion'].items() if 'infos' in k]
    # print(chunk_shapes)
    mode = "continuous" if chunk_shapes[0].count(',') == 2 else "processed"
    
    times = {k: v[0] for k, v in data['conversion'].items() if 'time' in k}

# NOTE: for processed file, the default chunks are the same as the full chunks
# don't repeat the default one
# the full chunks is (-1, -1, -1) and not (-1, -1, 1) as indicated (unavailable for processed files)
    if mode == "processed":
        for k in list(times.keys()):
            if (-1, -1, 1) == k[0]:
                del times[k]
            if True == k[0]:
                bkp = times[k]
                del times[k]
                times[('Full chunks',) + k[1:]] = bkp
    
    infos = {k: dict(v) for k, v in data['conversion'].items() if 'infos' in k}
    
    print(f'\tChunk shape & Memory order & Compressor & time (s) & size (disk)\\\\')
    print('\t\\hline')
    for params, time in times.items():
        try:
            params_info = infos[params[:-1] + ('infos',)]
        except KeyError:
            params_info = infos[((-1, -1, 1),) + params[1:-1] + ('infos',)]
        print(params_info)
        size_ram = params_info['No. bytes'].split(' ')[1][1:-1]
        size_disk = params_info['No. bytes stored'].split(' ')[1][1:-1]
        print(f'\t{params[0]} & {params[1]} & {params[2]} & {time:4.3} & {size_disk}\\\\')
    print(f'{size_ram = }')
