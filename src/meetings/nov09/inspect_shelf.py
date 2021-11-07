"inspect shelf: print a pretty dict of a list of shelf files"

import pathlib
import pprint
import shelve
import sys
import warnings


def load_shelf(path: str) -> dict:
    shelf = shelve.open(path, flag='r')
    return {k: shelf[k] for k in shelf.keys()}


def _run(path: str) -> None:
    pprint.pprint(load_shelf(path))


if __name__ == "__main__":
    for _file in sys.argv[1:]:
        _path = pathlib.Path(_file)
        if not _path.exists():
            warnings.warn(f'unable to find {_file}')
        if _path.is_dir():
            for _sub in _path.iterdir():
                if _sub.is_file():
                    _run(str(_sub))
        else:
            _run(str(_path))
