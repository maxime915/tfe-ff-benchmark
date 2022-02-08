"imzml 3d: get 3d info from an imzML file"

import argparse
import timeit
from typing import Callable

import zarr
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da

from src.utils.iter_chunks import iter_spatial_chunks


REDUCTION = {
    'min': np.amin,
    'max': np.amax,
    'mean': np.mean,
    'sum': np.sum,
    'median': np.median,
}


def reduce_img(
    zarr_path: str,
    reduction: str,
    mz_val: float,
    mz_tol: float,
    save_to: str = '',
    display: bool = True,
) -> None:
    """load a file, and show the TIC of all z planes

    - save_to: str, path to save the sublane to (appended by _z.png) [falsy
        value do not store]
    - show: request a call to matplotlib.pyplot.imshow"""

    if save_to == 'infer':
        save_to = zarr_path[:-5]

    intensities = zarr.open_group(zarr_path, mode='r')[0]
    mzs = zarr.open_group(zarr_path, mode='r').labels.mzs[0]

    shape = mzs.shape[-2:]

    path_stem = zarr_path.split('/')[-1]

    print(f'{reduction} for m/Z = {mz_val} +/- {mz_tol} for', end='\n\t')
    print(f'image {path_stem} of shape {shape}, and mode ', end='')

    # get the callable
    reduction_name = reduction
    reduction = REDUCTION[reduction]

    start = timeit.default_timer()

    if mzs.shape[-2:] == (1, 1):
        print('continuous')

        mz_band = mzs[:, 0, 0, 0]

        # search the unique mzs band
        low_i = np.searchsorted(mz_band, mz_val-mz_tol, side='left')
        high_i = np.searchsorted(mz_band, mz_val+mz_tol, side='right')

        data = intensities[
            low_i:high_i,  # m/Z
            0, :, :  # z,y,x
        ]

        img = reduction(data, 0)

    elif False:  # Uses too much memory
        print('processed - dask')

        intensities = da.from_zarr(zarr_path, '/0')
        mzs = da.from_zarr(zarr_path, '/labels/mzs/0')
        lengths = da.from_zarr(zarr_path, '/labels/lengths/0')

        def search_processed(mz_band, int_band, len_val) -> float:
            mz_band = mz_band[:len_val]
            low = np.searchsorted(mz_band, mz_val-mz_tol, side='left')
            high = 1 + np.searchsorted(mz_band, mz_val+mz_tol, side='right')
            return reduction(int_band[low:high])

        mzs_window = mzs[:, 0, :, :]
        int_window = intensities[:, 0, :, :]
        len_window = lengths[:, 0, :, :]

        img = da.apply_gufunc(search_processed, '(i),(i),()->()',
                              mzs_window, int_window, len_window,
                              axes=[(-3,), (-3,), (), ()],
                              allow_rechunk=True,
                              vectorize=True,
                              )

    elif True:  # chunk based access
        print('processed - chunk based')

        lengths: zarr.Array = zarr.open_group(zarr_path, mode='r').labels.lengths[0][:]

        img = np.zeros(shape)

        for chunk in iter_spatial_chunks(intensities):
            idx_4d = (slice(None),) + chunk
            idx_2d = chunk[1:]

            c_len = lengths[idx_4d]

            if not np.any(c_len):
                # indices = (idx, idx, ...)
                # -> make sure there is at least one element before loading
                #       the rest of the data
                continue

            c_int = intensities[idx_4d]
            c_mzs = mzs[idx_4d]

            buff = np.zeros(shape=c_int.shape[-2:], dtype=c_int.dtype)

            indices = np.nonzero(c_len)
            for (_, _, i, j) in zip(*indices):
                cap = c_len[0, 0, i, j]
                mz_band = c_mzs[:cap, 0, i, j]

                low_i = np.searchsorted(mz_band, mz_val-mz_tol, side='left')
                high_i = np.searchsorted(mz_band, mz_val+mz_tol, side='right')

                data = c_int[low_i:high_i, 0, i, j]

                buff[i, j] = reduction(data)

            img[idx_2d] = buff

    elif False:  # bad chunk access
        print('processed - naive access')

        lengths = zarr.open_group(zarr_path, mode='r').labels.lengths[0]

        img = np.zeros(shape)

        indices = np.nonzero(lengths)
        for idx, (_, _, i, j) in enumerate(zip(*indices)):
            cap = lengths[0, 0, i, j]
            mzs_band = mzs[:cap, 0, i, j]

            low_i = np.searchsorted(mzs_band, mz_val-mz_tol,
                                    side='left')
            high_i = np.searchsorted(mzs_band, mz_val+mz_tol,
                                         side='right')

            data = intensities[low_i:high_i, 0, i, j]

            img[i, j] = reduction(data)

    end = timeit.default_timer()

    print(f"reading done in {end-start}")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(f'{path_stem}\n{mz_val=} pm {mz_tol} ({reduction_name})')
    ax.imshow(img)

    plt.tight_layout()

    if len(save_to) > 0:
        fig.savefig(
            save_to + f'_{mz_val=}_pm{mz_tol}_{reduction_name}.png')

    if display:
        plt.show()


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()

    _parser.add_argument('--display', type=bool, default=False)
    _parser.add_argument('--save-to', type=str, default='',
                         dest='save_to', help="if empty, no image is saved")

    _parser.add_argument('--reduction', type=str, choices=REDUCTION.keys(),
                         default='sum', help="How to reduce a band to a scalar")

    _parser.add_argument('mz_val', type=float, help="center of the interval")
    _parser.add_argument('mz_tol', type=float,
                         help="half width of the interval")

    _parser.add_argument('zarr_path', type=str, help="path to the file")

    # import cProfile
    # import pstats

    # with cProfile.Profile() as pr:
    reduce_img(**vars(_parser.parse_args()))

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
