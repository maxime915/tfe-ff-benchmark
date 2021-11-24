"imzml 3d: get 3d info from an imzML file"

import argparse

import matplotlib.pyplot as plt
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser


def show_3d_planes(
        imzml_path: str, save_to: str = '', show: bool = True
        ) -> None:
    """load a file, and show the TIC of all z planes

    - save_to: str, path to save the sublane to (appended by _z.png) [falsy 
        value do not store]
    - show: request a call to matplotlib.pyplot.imshow"""

    parser = ImzMLParser(imzml_path, include_spectra_metadata='full')

    shape = (parser.imzmldict['max count of pixels y'],
            parser.imzmldict['max count of pixels x'])

    path_stem = imzml_path.split('/')[-1]

    # use a dict for all Z values
    #   to avoid having all images in memory at once, only save
    #   a list of indices in each plane
    planes = {}

    for idx, (coordinates, metadata) in enumerate(zip(
            parser.coordinates, parser.spectrum_full_metadata)):
        assert len(metadata.scans) == 1
        depth_plane = metadata.scans[0].user_params[2][3]

        if depth_plane not in planes:
            planes[depth_plane] = []

        planes[depth_plane].append((idx, coordinates))

    # draw the planes one by one
    for depth, coordinates in planes.items():
        img = np.zeros(shape)

        for idx, (x, y, _) in coordinates:
            _, intensities = parser.getspectrum(idx)
            img[y-1][x-1] = intensities.sum()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(f'{path_stem}\n{depth=}')
        ax.imshow(img)

        if len(save_to) > 0:
            fig.savefig(save_to + f'_{depth}.png')

        if show:
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--show', type=bool, default=True)
    _parser.add_argument('--save-to', type=str, default='', dest='save_to')
    _parser.add_argument('files', nargs='+', type=str)

    _args = _parser.parse_args()

    for _file in _args.files:
        show_3d_planes(_file, save_to=_args.save_to, show=_args.show)
