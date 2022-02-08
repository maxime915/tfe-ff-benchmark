"imzml 3d: get 3d info from an imzML file"

import argparse
import pathlib

import zarr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from .iter_chunks import iter_loaded_chunks

def build_tic_data(zarr_path: str) -> np.ndarray:
    
    # open intensities
    z = zarr.open_group(zarr_path, mode='r')[0]
    
    img = np.zeros(shape=z.shape[-2:], dtype=z.dtype)
    
    for chunk in iter_loaded_chunks(z, skip=2):
        img[chunk] = z[:, 0, chunk[0], chunk[1]].sum(axis=0)

    return img

_INFER = object()

def show_tic_planes(
    zarr_path: str,
    save_to: str = _INFER,
    show: bool = True
) -> None:
    """load a file, and show the TIC of all z planes

    - save_to: str, path to save the sublane to (appended by _z.png) [falsy 
        value do not store]
    - show: request a call to matplotlib.pyplot.imshow"""

    img = build_tic_data(zarr_path)
    
    # normalization
    img = np.log(img)
    
    path = pathlib.Path(zarr_path)

    fig, ax = plt.subplots(1, 1)
        
    c = ax.pcolor(img)
    fig.colorbar(c, ax=ax)

    ax.set_title(path.stem)
    ax.imshow(img, interpolation=None)
    
    plt.tight_layout()
    
    if save_to == _INFER:
        save_to = str(path.with_suffix('')) + '_TIC'

    if save_to:
        if not isinstance(save_to, pathlib.Path):
            save_to = pathlib.Path(save_to)
        if save_to.suffix not in ['.png', '.jpeg', '.tiff', '.tif', '.jpg']:
            save_to = save_to.with_suffix('.png')
        save_to = str(save_to)
        fig.savefig(save_to)

    if show:
        plt.show()
            

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--show', type=bool, default=True)
    _parser.add_argument('--save-to', type=str, default=_INFER, dest='save_to')
    _parser.add_argument('file', type=str)

    _args = vars(_parser.parse_args())
    
    show_tic_planes(_args.pop('file'), **_args)
