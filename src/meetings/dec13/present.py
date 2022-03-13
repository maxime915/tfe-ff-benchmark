"present: script to show a graph of the different access types"

import argparse
from collections import OrderedDict
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .inspect_db import DB

if __name__ != "__main__":
    sys.exit("should not be imported")

import enum

class Command(enum.Enum):
    BAND = enum.auto()
    TIC = enum.auto()
    SEARCH = enum.auto()
    OVERLAP = enum.auto()
    CONVERSION = enum.auto()
    COMPRESSOR = enum.auto()

    @staticmethod
    def str_choices() -> List[str]:
        return [choice.name.lower() for choice in Command]
    
    @staticmethod
    def parse(text: str) -> 'Command':
        mapper = {choice.name.lower() : choice for choice in Command}
        
        if text.lower() in mapper:
            return mapper[text.lower()]
        
        raise ValueError(f'invalid Command: {text}')

parser = argparse.ArgumentParser()
parser.add_argument('command', choices=['all'] + Command.str_choices(), help="command to run")
parser.add_argument('file', type=str, help='shelf store file')

args = parser.parse_args()

data = OrderedDict(sorted(DB(args.file).load_all().items(), key=str))

if 'all' == args.command:
    commands = list(Command)
else:
    commands = [Command.parse(args.command)]

def chunk_to_str(chunk) -> str:
    if not isinstance(chunk, str):
        chunk = str(chunk)
    if chunk == '(1, 1, -1, -1)':
        return 'flat'
    if chunk == '(-1, 1, 1, 1)':
        return 'deep'
    if chunk == 'True':
        return 'auto'
    return chunk

if Command.BAND in commands:
    time_dict = {k: v for k, v in data.items() if 'band' in k}
    imzml_number = data[('benchmark infos',)]['imzml number']
    zarr_number = data[('benchmark infos',)]['zarr number']
    
    continuous_mode = data[('imzml info',)].continuous_mode
    mode = "continuous" if continuous_mode else "processed"
        
    # load imzML as a reference
    imzml = np.array(data[('imzml_raw', 'band')]) / imzml_number

    # load all Zarr variation
    zarr_dict = {k: v for k, v in time_dict.items() if 'imzml_zarr' in k}
 
    # make a bar plot
    names = ['imzml']
    for k in zarr_dict.keys():
        name = [chunk_to_str(k[1])]
        for i in k[2:-1]:
            name.append(str(i))
        names.append('\n'.join(name))
    
    zarr_arrays = [np.array(v) / zarr_number for v in zarr_dict.values()]

    means = [imzml.mean()] + [a.mean() for a in zarr_arrays]
    errs = [imzml.std()] + [a.std() for a in zarr_arrays]

    if False and continuous_mode:
        names = names[1:]
        means = means[1:]
        errs = errs[1:]
    
    fig = plt.figure(figsize=[10.4, 4.8])
    axis = fig.add_subplot(111)

    axis.bar(names, means, yerr=errs, width=0.8)

    axis.set_yscale('log')
    axis.set_ylim(5e-3, 2)

    limit_high = axis.get_ylim()[1]
    for i, val in enumerate(means):
        if val > limit_high:
            axis.text(i - 0.25, limit_high - 0.5, f"{val:.2f}")

    axis.set_title(f'Spectral Access time (single band) - {mode} mode file')
    axis.set_ylabel('Mean time (s)')
    
    fig.tight_layout()
    fig.savefig(f'band_{mode}.png')
    

if Command.TIC in commands:
    time_dict = {k: v for k, v in data.items() if 'tic' in k}
    imzml_number = data[('benchmark infos',)]['imzml number']
    zarr_number = data[('benchmark infos',)]['zarr number']
    
    continuous_mode = data[('imzml info',)].continuous_mode
    mode = "continuous" if continuous_mode else "processed"
    
    for window_width in (16, 32, 64):    
            
        # load imzML as a reference
        imzml = np.array(data[('imzml_raw', 'tic', (window_width, window_width))]) / imzml_number

        # load all Zarr variation
        zarr_dict = {k: v for k, v in time_dict.items() if (window_width, window_width) in k and 'imzml_zarr' in k }

        # make a bar plot
        names = ['imzml']
        for k in zarr_dict.keys():
            name = [chunk_to_str(k[1])]
            for i in k[2:-2]:
                name.append(str(i))
            names.append('\n'.join(name))
        
        zarr_arrays = [np.array(v) / zarr_number for v in zarr_dict.values()]
        
        means = [imzml.mean()] + [z.mean() for z in zarr_arrays]
        yerrs = [imzml.std()] + [z.std() for z in zarr_arrays]

        if False and continuous_mode:
            names = names[1:]
            means = means[1:]
            errs = errs[1:]
        
        fig = plt.figure(figsize=[10.4, 4.8])
        axis = fig.add_subplot(111)
        
        axis.bar(names, means, yerr=yerrs, width=0.8)

        axis.set_yscale('log')
        axis.set_ylim(1e-2, 3)
        
        limit_high = axis.get_ylim()[1]
        for i, val in enumerate(means):
            if val > limit_high:
                axis.text(i - 0.25, limit_high - 0.5, f"{val:.2f}")
        
        axis.set_title(f'Channel Sum time (sum over all bands) - {mode} mode file')
        axis.set_ylabel('Mean time (s)')
        
        fig.tight_layout()
        fig.savefig(f'tic_{mode}_{window_width}.png')

if Command.SEARCH in commands:
    time_dict = {k: v for k, v in data.items() if 'search' in k}
    imzml_number = data[('benchmark infos',)]['imzml number']
    zarr_number = data[('benchmark infos',)]['zarr number']
    
    continuous_mode = data[('imzml info',)].continuous_mode
    mode = "continuous" if continuous_mode else "processed"
    
    for window_width in (16, 32, 64):    
            
        # load imzML as a reference
        imzml = np.array(data[('imzml_raw', 'search', (window_width, window_width))]) / imzml_number

        # load all Zarr variation
        zarr_dict = {k: v for k, v in time_dict.items() if (window_width, window_width) in k and 'imzml_zarr' in k }

        # make a bar plot
        names = ['imzml']
        for k in zarr_dict.keys():
            name = [chunk_to_str(k[1])]
            for i in k[2:-2]:
                name.append(str(i))
            names.append('\n'.join(name))
        
        zarr_arrays = [np.array(z) / zarr_number for z in zarr_dict.values()]
        
        means = [imzml.mean()] + [z.mean() for z in zarr_arrays]
        errs = [imzml.std()] + [z.std() for z in zarr_arrays]
        
        # remove raw imzml
        if False:
            names = names[1:]
            values = values[1:]
        
        fig = plt.figure(figsize=[10.4, 4.8])
        axis = fig.add_subplot(111)
        
        axis.bar(names, means, yerr=errs, width=0.8)

        axis.set_yscale('log')
        axis.set_ylim(1e-2, 2)
        
        limit_high = axis.get_ylim()[1]
        for i, val in enumerate(means):
            if val > limit_high:
                axis.text(i - 0.25, limit_high - 0.5, f"{val:.2f}")

        axis.set_title(f'Channel Search time - {mode} mode file ({window_width})')
        axis.set_ylabel('Mean time (s)')
        
        fig.tight_layout()
        fig.savefig(f'search_{mode}_{window_width}.png')

if Command.OVERLAP in commands:
    time_dict = {k: v for k, v in data.items() if 'tic-overlap' in k}
    
    continuous_mode = data[('imzml info',)].continuous_mode
    mode = "continuous" if continuous_mode else "processed"

    # only do window of 16x16: this is the only one supported by both files
    # compute ratio overlap11 / overlap00
    for window_width in [16, 32, 64]:
        window = (window_width, window_width)

        overlap_00_dict = {k[1:4]: v for k, v in time_dict.items(
        ) if k[-3:] == ('tic-overlap', window, (0, 0))}
        overlap_11_dict = {k[1:4]: v for k, v in time_dict.items(
        ) if k[-3:] == ('tic-overlap', window, (1, 1))}

        names = []
        values = []
        for (k, o00) in overlap_00_dict.items():
            o11 = overlap_11_dict[k]

            name = [chunk_to_str(k[0])]
            for i in k[1:]:
                name.append(str(i))
            names.append('\n'.join(name))

            # no need to /zarr_number : they have the same amount of trials
            values.append(np.array(o11).mean() / np.array(o00).mean())
        
        if not values:
            continue

        fig = plt.figure(figsize=[10.4, 4.8])
        axis = fig.add_subplot(111)

        axis.bar(names, values, width=0.8)

        axis.set_yscale('log')
        axis.set_ylim(1, 4)

        limit_high = axis.get_ylim()[1]
        for i, val in enumerate(values):
            if val > limit_high:
                axis.text(i - 0.25, limit_high - 0.5, f"{val:.2f}")

        axis.set_title(
            f'Overlap time ratio on a window of size {window_width} - {mode} mode file')
        axis.set_ylabel('Time ratio')

        fig.tight_layout()
        fig.savefig(f'ticoverlap_{mode}_{window_width}.png')

if Command.COMPRESSOR in commands:
    zarr_number = data[('benchmark infos',)]['zarr number']
    time_dict = {k: v for k, v in data.items() if 'imzml_zarr' in k and 'infos' not in k and 'conversion time' not in k}
    no_compression = {k[:2] + k[3:]: v for k, v in time_dict.items() if k[2] == None}
    default_compression = {k[:2] + k[3:]: v for k, v in time_dict.items() if k[2] == 'default'}

    keys = sorted(no_compression.keys(), key=str)
    assert sorted(default_compression.keys(), key=str) == keys

    deltas = {
        k: (np.mean(np.array(default_compression[k]) / zarr_number) - 
            np.mean(np.array(no_compression[k]) / zarr_number))
        for k in keys
    }

    print('\n'.join(f"{k}: {v}" for k, v in deltas.items()))


if Command.CONVERSION in commands:
    time_dict = {k: v for k, v in data.items() if 'conversion time' in k}
    imzml_number = data[('benchmark infos',)]['conversion number']

    print(f'\tChunk shape & Compressor & Memory order & time (s) & size (disk)\\\\')
    print('\t\\hline')
    for params, time in time_dict.items():
        key = params[:4] + ('infos', 'intensities')
        params_info = dict(data[key])
        
        chunks = params[1]
        if chunks == True:
            # auto chunk -> guess the value
            chunks = params_info['Chunk shape']

        size_ram = params_info['No. bytes'].split(' ')[1][1:-1]
        size_disk = params_info['No. bytes stored'].split(' ')[1][1:-1]
                
        print(f'\t{chunks} & {params[2]} & {params[3]} & {np.mean(time):4.3} & {size_disk}\\\\')

    # # see what was the default chunk shape
    # chunk_shapes = [dict(v)['Chunk shape'] for k, v in data['conversion'].items() if 'infos' in k]
    # # print(chunk_shapes)
    # mode = "continuous" if data['imzml info'].continuous_mode else "processed"
    
    # times = {k: v[0] for k, v in data['conversion'].items() if 'time' in k}
    
    # infos = {k: dict(v) for k, v in data['conversion'].items() if 'infos' in k}
    
    # print(f'\tChunk shape & Compressor & Memory order & time (s) & size (disk)\\\\')
    # print('\t\\hline')
    # for params, time in times.items():
    #     try:
    #         params_info = infos[params[:-1] + ('infos',)]
    #     except KeyError:
    #         params_info = infos[((-1, -1, 1),) + params[1:-1] + ('infos',)]
    #     print(params_info)
    #     size_ram = params_info['No. bytes'].split(' ')[1][1:-1]
    #     size_disk = params_info['No. bytes stored'].split(' ')[1][1:-1]
    #     print(f'\t{params[0]} & {params[1]} & {params[2]} & {time:4.3} & {size_disk}\\\\')
    # print(f'{size_ram = }')
