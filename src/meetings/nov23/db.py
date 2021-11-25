"db: create a DB class "

import datetime
import codecs
import pathlib
import pickle
import shelve
import string
import typing


_ALPHA_NUM = string.ascii_letters + string.digits


def get_db_dir(file_or_dir: str = __file__, name : str = 'results') -> pathlib.Path:
    """get a directory suitable to contain the db

    - if file_or_dir is a dir, a sub-dir 'results' will be created & returned
    - else, a sibling dir 'results' will be created & returned
    
    - name: directory name
    """

    path = pathlib.Path(file_or_dir).resolve()
    if path.is_file():
        path = path.parent

    db_dir = path / name
    if not db_dir.exists():
        db_dir.mkdir()

    return db_dir


class DB:
    """wrapper around shelve.Shelf avoiding to keep the shelf open for too long.
    Parallel access to the same shelf my break depending on the platform (see
    https://docs.python.org/3/library/shelve.html#restrictions)

    DB uses base64 encoding of pickled keys to allow more object as keys."""

    def __init__(self, path: str) -> None:
        self.path = path

    def save_val_at(self, value, *keys: typing.List[str], overwrite=False) -> None:
        "save a value into the shelf with the given key"

        key = codecs.encode(pickle.dumps(keys), "base64").decode()

        with shelve.open(self.path, flag='c') as shelf:
            if key in shelf and not overwrite:
                raise KeyError(f'cannot overwrite key: {key}')
            shelf[key] = value

    def load(self, *keys: typing.List[str]) -> typing.Any:
        "load a value from the shelf with a given key"

        key = codecs.encode(pickle.dumps(keys), "base64").decode()

        with shelve.open(self.path, flag='r') as shelf:
            return shelf[key]

    def load_all(self) -> dict:
        "returns a dict containing all the data in the db"

        result = {}
        with shelve.open(self.path, flag='r') as shelf:
            for key_raw, value in shelf.items():
                key = pickle.loads(codecs.decode(key_raw.encode(), "base64"))
                result[key] = value

        return result


def get_db_for_file(file: str, db_dir: pathlib.Path = None) -> DB:
    if db_dir is None:
        db_dir = get_db_dir()
    if isinstance(db_dir, str):
        db_dir = get_db_dir(db_dir)

    now = datetime.datetime.now()
    key = ''.join(l for l in file + str(now) if l in _ALPHA_NUM)
    if not key:
        raise ValueError(f"{file=} must contain alphanumeric characters")

    db = DB(str(db_dir / key))

    db.save_val_at({
        'file': file,
        'date': now,
        'key': key,
    }, 'metadata')

    return db
