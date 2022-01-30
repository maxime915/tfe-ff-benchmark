"inspect_db: show information on a result file"

import argparse
import collections
import re

from typing import Optional

from .db import DB


def inspect(db_path: str, full: bool = False, filter: Optional[re.Pattern] = None) -> None:
    ""
    db = DB(db_path)
    try:
        content = collections.OrderedDict(sorted(db.load_all().items()))
    except Exception:
        content = collections.OrderedDict(
            sorted(db.load_all().items(), key=str))
    for key, value in content.items():
        if filter:
            if not filter.match(str(key)):
                continue
        if full:
            print(f'db[{key}] = {value}')
        else:
            print(key)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('files', type=str, nargs='+')
    _parser.add_argument('--full', action='store_true', dest='full')
    _parser.add_argument('--filter', type=str, default='')
    _parser.set_defaults(full=False)

    _args = _parser.parse_args()

    for _file in _args.files:
        inspect(_file, _args.full, re.compile(
            _args.filter) if _args.filter else None)
