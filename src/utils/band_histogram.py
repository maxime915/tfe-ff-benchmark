"band_histogram: "

import argparse
import warnings

from typing import List, Literal
from pyimzml.ImzMLParser import ImzMLParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def get_lengths(imzml: str) -> List[int]:
    "get the sorted lengths for one file"

    warnings.filterwarnings('ignore', r'Accession I?MS:')

    parser = ImzMLParser(imzml, ibd_file=None)
    
    pixel_count = parser.imzmldict['max count of pixels x'] * \
        parser.imzmldict['max count of pixels y']
    
    return sorted(parser.mzLengths), pixel_count


_SAVE_TO_INFER = object()


def show_graph(
    *imzml: str,
    show: bool = False,
    save_to: str = _SAVE_TO_INFER,
):

    # fetch the data in parallel
    pool = Pool()
    bands_per_file = pool.map(get_lengths, imzml)

    fig, axis = plt.subplots(1, 1, figsize=[10.4, 4.8])

    axis.set_yscale('log')
    axis.set_xscale('log')
    
    axis.set_ylabel('Frequency')
    axis.set_xlabel('Spectral Length')
    
    for file, (bands, p_count) in zip(imzml, bands_per_file):
        # https://stackoverflow.com/a/42884659

        bands = np.array(bands)
        std = bands.std()
        
        bins = 2 if std == 0 else 'auto'
                
        values, edges = np.histogram(bands, bins=bins)
        centers = 0.5*(edges[1:] + edges[:-1])
    
        p_count_str = f" ({p_count:.1E} pixels)"
        axis.plot(centers, values, '-', label=Path(file).stem + p_count_str)

    axis.legend()
    axis.set_title('Frequencies of the spectral lengths')

    plt.tight_layout()
    
    if save_to is _SAVE_TO_INFER:
        save_to = "tmp.png"
    
    if save_to:
        fig.savefig(save_to)
    
    if show:
        plt.show()


if __name__ == "__main__":
    _arg_parser = argparse.ArgumentParser()
    _arg_parser.add_argument('imzml', type=str, nargs='+')
    _arg_parser.add_argument('--show', type=bool, default=False)
    _arg_parser.add_argument('--save_to', type=str, default=_SAVE_TO_INFER)

    _args = vars(_arg_parser.parse_args())
    _files = _args.pop('imzml')

    show_graph(*_files, **_args)
