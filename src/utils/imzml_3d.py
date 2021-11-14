"imzml 3d: get 3d info from an imzML file"

import argparse

import matplotlib.pyplot as plt
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser


def show_3d_planes(imzml_path: str, save_to: str = '', show: bool = True) -> None:
    """load a file, and show the TIC of all z planes

    - save_to: str, path to save the sublane to (appended by _z.png) [falsy value do not store]
    - show: request a call to matplotlib.pyplot.imshow"""

    parser = ImzMLParser(imzml_path)

    shape = (parser.imzmldict['max count of pixels y'],
             parser.imzmldict['max count of pixels x'])

    path_stem = imzml_path.split('/')[-1]

    def show_subplane(start, stop, z=0):

        img = np.zeros(shape)

        last_x = parser.coordinates[start][0]
        next_start = stop

        for i in range(start, stop):
            x, y, _ = parser.coordinates[i]

            if x < last_x:
                next_start = i
                break

            _, intensities = parser.getspectrum(i)
            img[y-1, x-1] = intensities.sum()

            last_x = x

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(f'{path_stem} {z=}')
        ax.imshow(img)

        if len(save_to) > 0:
            fig.savefig(save_to + f'_{z}.png')

        if show:
            plt.show()

        if next_start < stop:
            show_subplane(next_start, stop, z+1)

    show_subplane(0, len(parser.coordinates))


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--show', type=bool, default=True)
    _parser.add_argument('--save-to', type=str, default='', dest='save_to')
    _parser.add_argument('files', nargs='+', type=str)

    _args = _parser.parse_args()

    for _file in _args.files:
        show_3d_planes(_file, save_to=_args.save_to, show=_args.show)
