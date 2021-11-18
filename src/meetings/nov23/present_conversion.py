"present: script to show a graph of the different access types"

import argparse
import collections
import sys

import matplotlib.pyplot as plt
import numpy as np

from .inspect_db import DB

if __name__ != "__main__":
    sys.exit("should not be imported")

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='shelf store file')

args = parser.parse_args()

# fuse all files (overriding older)
data = DB(args.file).load_all()

# representation of each size
size_to_str = {
    1073741824: "1GiB",
    2147483648: '2GiB',
    4294967296: '4GiB',
}

for size in size_to_str.keys():

    # select matching keys
    sample = {k: v for k, v in data.items() if size in k}
    times = {k: v for k, v in sample.items() if 'conversion time' in k}

    # build a figure for this size
    #   select one color for the rechunker attribute
    #   plot a bar plot with doubled bars to show rechunker attribute

    # unique keys
    names = {k[:3] for k in times.keys()}
    # sorted and list
    names = sorted(names, key=str)
    # filter chunks: replace 'True' by the actual value
    for i, name in enumerate(names):
        if name[0] != True:
            continue
        # find chunks
        chunks = sample[(*name, False, size, 'infos', 'intensities')][3][1]
        # replace the value in the list
        names[i] = (chunks, *name[1:])

    # get each name as a multiline label
    names = ['\n'.join(str(e) for e in name) for name in names]
    
    # get the data for each bar
    times_left = {k: v for k, v in times.items() if k[3] == True}
    times_right = {k: v for k, v in times.items() if k[3] == False}

    # sort it
    times_left = collections.OrderedDict(sorted(times_left.items(), key=str))
    times_right = collections.OrderedDict(sorted(times_right.items(), key=str))

    # select time only
    times_left = [np.mean(repeat) for repeat in times_left.values()]
    times_right = [np.mean(repeat) for repeat in times_right.values()]

    # prep plot
    x = np.arange(len(names))
    width = 0.35

    fig = plt.figure(figsize=[10.4, 4.8])
    axis = fig.add_subplot(1, 1, 1)

    # rechunker to the left, other to the right
    rect_left = axis.bar(x - width/2, times_left, width, label='rechunker')
    rect_right = axis.bar(x + width/2, times_right, width, label='zarr copy')

    # pretty plot
    axis.set_title(f'Conversion benchmark on a processed file (threshold={size_to_str[size]})')
    axis.set_ylabel('time (s)')
    axis.set_xticks(x)
    axis.set_xticklabels(names)
    axis.legend()

    # optional: add exact value
    # axis.bar_label(rect_left, padding=3)
    # axis.bar_label(rect_right, padding=3)
    
    fig.tight_layout()
    plt.show()

    
# access_number = data['parameters']['access_number']

# # see what was the default chunk shape

# chunk_shapes = [dict(v)['Chunk shape'] for k, v in data['conversion'].items() if 'infos' in k]
# # print(chunk_shapes)

# mode = "continuous" if chunk_shapes[0].count(',') == 2 else "processed"

# # load imzML as a reference
# imzml = np.array(data['imzml_raw']['band']) / access_number

# # load all Zarr variation
# zarr_dict : dict = data['imzml_zarr']
# zarr_dict = { k: v for k, v in zarr_dict.items() if 'band' in k }

# # make a bar plot
# names = ['imzml'] + ['\n'.join(str(i) for i in k[:-1]) for k in zarr_dict.keys()]
# values = [imzml.mean()] + [np.array(v).mean() / access_number for v in zarr_dict.values()]

# fig = plt.figure(figsize=[10.4, 4.8])
# axis = fig.add_subplot(111)

# axis.bar(names, values, width=0.8)
# axis.set_yscale('log')
# axis.set_title(f'Spectral Access time (single band) - {mode} mode file')
# axis.set_ylabel('Mean time (s)')

# fig.tight_layout()
# fig.savefig(f'band_{mode}.png')
