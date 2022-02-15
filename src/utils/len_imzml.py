"imzml 3d: get 3d info from an imzML file"

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser


def show_tic_planes(
        imzml_path: str, save_to: str = '', show: bool = True
        ) -> None:
    """load a file, and show the TIC of all z planes

    - save_to: str, path to save the sublane to (appended by _z.png) [falsy 
        value do not store]
    - show: request a call to matplotlib.pyplot.imshow"""

    print(f'opening file {imzml_path}')

    parser = ImzMLParser(imzml_path, ibd_file=None)

    shape = (parser.imzmldict['max count of pixels y'],
            parser.imzmldict['max count of pixels x'])

    path_stem = imzml_path.split('/')[-1][:-6]

    mode = 'continuous' if 'continuous' in parser.metadata.file_description.param_by_name else 'processed'

    print(f'image {path_stem} of shape {shape}, type {mode}')

    # draw the planes one by one
    img = np.zeros(shape, dtype=int)

    for idx, (x, y, _) in enumerate(parser.coordinates):
        img[y-1][x-1] = parser.mzLengths[idx]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    c = ax.pcolor(img)
    fig.colorbar(c, ax=ax)

    ax.set_title(path_stem)
    ax.imshow(img, interpolation=None)

    plt.tight_layout()

    if len(save_to) > 0:
        fig.savefig(save_to + '.png')

    if show:
        plt.show()
            

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--show', type=bool, default=True)
    _parser.add_argument('--save-to', type=str, default='', dest='save_to')
    _parser.add_argument('files', nargs='+', type=str)

    _args = _parser.parse_args()

    warnings.filterwarnings('ignore', message=r'Accession I?MS')

    for _file in _args.files:
        show_tic_planes(_file, save_to=_args.save_to, show=_args.show)
