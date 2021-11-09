"list_imzml: present summarized information on imzML files"

import pathlib
import sys
import time
import warnings

from ...benchmark.imzml import imzml_path_to_info

if __name__ != "__main__":
    sys.exit("should not be imported")

def sizeof_fmt(num, suffix="B"):
    "from https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size"
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def _run(path: pathlib.Path, rec=True) -> None:
    if path.is_dir():
        for _sub in path.iterdir():
            _run(_sub, rec=False)  # 1 level deep only
    elif path.suffix == '.imzML':
        start = time.time()
        infos = imzml_path_to_info(str(path))
        ibd = path.with_suffix('.ibd')
        sizes = (path.stat().st_size, ibd.stat().st_size)
        end = time.time()

        print(infos)
        print(path.stem)
        print(f'\tcontinuous mode: {infos.continuous_mode}')
        print(f'\tshape (x, y, z): {infos.shape}', end='')
        if infos.continuous_mode:
            print(f' \tlen(m/Z): {infos.band_size_min}')
        else:
            print(f' \tlen(m/Z): [{infos.band_size_min}, {infos.band_size_max}]')
        print(f'\tm/Z \\in [{infos.mzs_min}, {infos.mzs_max}]')
        print(f'\timzML size: {sizeof_fmt(sizes[0])} \t| \tibd size: {sizeof_fmt(sizes[1])}')
        print(f'\ttook {end - start}s to analyze')
        print('')


warnings.filterwarnings('ignore')

for _file in sys.argv[1:]:
    _run(pathlib.Path(_file))
