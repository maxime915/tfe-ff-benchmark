""

import pathlib
import sys

import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt

from ..utils.iter_chunks import iter_loaded_chunks

files = sys.argv[1:]

mz_table = pd.read_csv("mz value + lipid name.csv", sep=";")

# FIXME TODO REMOVE
# mz_table = mz_table[mz_table.Name.isin(['DPPC', 'PAzPC',  'PLPC', 'SLPC', 'PAPC', 'PAPC-OH'])]
# TODO there may be a way to parallelize searchsorted in order to find multiple values at once

def processed_search(z: zarr.Group, mz_val: float, mz_tol: float) -> np.ndarray:

    ints = z["/0"]
    lengths = z["/labels/lengths/0"]
    mzs = z["/labels/mzs/0"]

    img = np.zeros(shape=ints.shape[2:], dtype=ints.dtype)

    for (cy, cx) in iter_loaded_chunks(ints, skip=2):
        int_window = ints[:, 0, cy, cx]
        lengths_window = lengths[0, 0, cy, cx]
        mzs_window = mzs[:, 0, cy, cx]

        for i in range(int_window.shape[1]):
            for j in range(int_window.shape[2]):
                len = lengths_window[i, j]
                mz_band = mzs_window[:len, i, j]
                low = np.searchsorted(mz_band, mz_val - mz_tol, side="left")
                hi = np.searchsorted(mz_band, mz_val + mz_tol, side="right")

                if low == hi:
                    continue

                img[i + cy.start, j + cx.start,] = int_window[
                    low:hi, i, j
                ].sum(axis=0)
    return img


def continuous_search(z: zarr.Group, mz_val: float, mz_tol: float) -> np.ndarray:

    ints = z["/0"]
    mzs = z["/labels/mzs/0"]

    img = np.zeros(shape=ints.shape[2:], dtype=ints.dtype)

    mz_band = mzs[:, 0, 0, 0]
    low = np.searchsorted(mz_band, mz_val - mz_tol, side="left")
    hi = np.searchsorted(mz_band, mz_val + mz_tol, side="right")

    if hi == 0:
        print(f'{low=} - {hi=}')
        print(f'{mz_val=} {mz_tol=}')
        print(f'{mz_band.min()=} {mz_band.max()=}')
    
    hi = max(hi, low+1)

    print(f'{low=} - {hi=}')

    if low == hi:
        return img

    for (cy, cx) in iter_loaded_chunks(ints, skip=2):
        int_window = ints[low:hi, 0, cy, cx]
        img[cy, cx] = int_window.sum(axis=0)

    return img


def save_lipid(z_path: str, mz_val: float, mz_tol: float, name: str) -> None:
    z = zarr.open_group(z_path, mode="r")
    if z.attrs["pims-msi"]["binary_mode"] == "continuous":
        img = continuous_search(z, mz_val, mz_tol)
    else:
        img = processed_search(z, mz_val, mz_tol)
    
    dpi = 300

    fig = plt.figure(dpi=dpi)
    axis = fig.add_subplot(111)

    source = pathlib.Path(z_path)
    stem = source.stem
    title = stem + "-" + name

    slug = "".join(c for c in name.lower() if c not in ["/", " "])
    destination = str(source.with_suffix("")) + f"_extract_{mz_val}_{slug}.png"

    c = axis.pcolor(img)
    fig.colorbar(c, ax=axis)

    axis.set_title(title)
    axis.imshow(img, interpolation='none')

    plt.tight_layout()
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)



def process_file_list(file_list):
    for file in file_list:
        process_file(file)


def process_file(file):
    for row in mz_table.itertuples():
        mz_val, mz_tol, name = row[1:4]
        print(f'{file=} {name=}')
        save_lipid(file, mz_val, mz_tol, name)

process_file_list(files)

"""
NOTE: line 17 altered: (removed the f)
'524.37107;0.00114f;LysoSPC/  PAF -O-16:0;' -> '524.37107;0.00114;LysoSPC/  PAF -O-16:0;'
"""
