"imzml 3d: get 3d info from an imzML file"

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def get_data(base_chunk, min_chunk, max_chunk, bsize) -> np.ndarray:
    data = base_chunk * (2**np.log10(bsize/(1024.*1024)))

    data = np.minimum(data, max_chunk)
    data = np.maximum(data, min_chunk)

    return data

_INFER = object()

def show_tic_planes(
    save_to: str = _INFER,
    show: bool = True
) -> None:

    total_size = np.logspace(10, 50, base=2)
    data_h = get_data(
        16 * 1024,
        8 * 1024,
        1024 * 1024,
        total_size
    )
    data_z = get_data(
        256 * 1024,
        128 * 1024,
        64 * 1024 * 1024,
        total_size
    )
    data_d = 128 * 2**20 + np.zeros_like(total_size)

    fig, ax = plt.subplots(1, 1)
        
    ax.set_title('Target chunk size comparison between Zarr and H5py')

    ax.plot(total_size, data_z, label='zarr')
    ax.plot(total_size, data_h, label='h5py')
    # ax.plot(total_size, data_d, label='dask')
    
    ax.axvline(x=1763704832, linestyle='--', label='240_s512_s512')
    ax.axvline(x=12687769600, linestyle='-.', label='190225_g06_right_s512_s512')
    ax.axvline(x=1587837394304, linestyle=':', label='S80CTa Region 013')
    
    ax.set_xlabel('Total size of the array (bytes)')
    ax.set_ylabel('Target size of the chunk (bytes)')
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    ax.legend()
    
    plt.tight_layout()
    
    if save_to == _INFER:
        save_to = 'comparison.png'

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

    _args = vars(_parser.parse_args())
    
    show_tic_planes(**_args)
