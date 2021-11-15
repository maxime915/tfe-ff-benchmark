"inspect_db: show information on a result file"

import argparse
import collections

from .db import DB


def inspect(db_path: str, full: bool = False) -> None:
    ""
    db = DB(db_path)
    content = collections.OrderedDict(sorted(db.load_all().items()))
    for key, value in content.items():
        if full:
            print(f'db[{key}] = {value}')
        else:
            print(key)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('files', type=str, nargs='+')
    _parser.add_argument('--full', action='store_true', dest='full')
    _parser.set_defaults(full=False)

    _args = _parser.parse_args()

    for _file in _args.files:
        inspect(_file, _args.full)
