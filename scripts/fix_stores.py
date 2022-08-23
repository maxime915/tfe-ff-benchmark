"inspect all svelte store to find useful information"

import binascii
import dbm
import sys

# import src to unpickle some results
sys.path.insert(0, ".")
import src

import shelve
import codecs
import pickle
import json
import os

_FIXED_SUFFIX = ".bkp.json"


def get_shelve_path_list(source_dir: str, match: str = "results"):
    results = []
    for root, _, files in os.walk(source_dir):
        if root.endswith(match):
            results.extend([root + "/" + f for f in files if "." not in f])
    return results


def get_backup_file_list(source_dir: str, match: str = _FIXED_SUFFIX):
    results = []
    for root, _, files in os.walk(source_dir):
        results.extend([root + "/" + f for f in files if f.endswith(match)])
    return results


def make_json(shelve_path: str, json_path: str):
    
    try:
        db = shelve.open(shelve_path)
    except dbm.error[0] as e:
        raise ValueError("unable to open shelf") from e
    
    with db:
        assert type(db) == shelve.DbfilenameShelf

        keys = list(db.keys())
        serializable_dict = {}

        for key in keys:
            try:
                raw_key = pickle.loads(codecs.decode(key.encode(), "base64"))
            except (pickle.UnpicklingError, binascii.Error) as e:
                raw_key = key
            str_key = str(raw_key)

            # a python object, consisting of list, tuple, dict, str, float
            val = db[key]
            # tuple are not supported by JSON: store a pickled and repr version of val
            val_str = str(val)
            val_enc = codecs.encode(pickle.dumps(val), "base64").decode("utf8")

            serializable_dict[key] = dict(
                str_key=str_key, val_str=val_str, val_enc=val_enc
            )

    with open(json_path, "w", encoding="utf8") as json_file:
        json.dump(serializable_dict, json_file)


def make_shelve(json_path: str, shelve_path: str):

    with open(json_path, "r", encoding="utf8") as json_file:
        serialization_dict = json.load(json_file)

    if shelve_path is None:
        db = shelve.Shelf({})
    else:
        db = shelve.open(shelve_path)

    with db:
        for key, value_dict in serialization_dict.items():
            # key is the original key, all is fine

            # val should be recovered from val_enc
            val_enc: str = value_dict["val_enc"]
            val = pickle.loads(codecs.decode(val_enc.encode("utf8"), "base64"))

            db[key] = val

    # should be good to go...


def make_backups():
    files_to_fix = get_shelve_path_list("src/meetings")

    for file in files_to_fix:
        make_json(file, file + _FIXED_SUFFIX)
        make_shelve(file + _FIXED_SUFFIX, None)


def restore_all_shelves():

    backup_files = get_backup_file_list("src/meetings")

    for file in backup_files:
        if not file.endswith(_FIXED_SUFFIX):
            raise ValueError(f"bad suffix encountered in {file=!r}")
        
        base = file[:-len(_FIXED_SUFFIX)]
        make_shelve(file, base + ".TO_DELETE")

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("command", choices=["make-backup", "restore-shelves"])
    
    args = parser.parse_args()
    if args.command == "make-backup":
        make_backups()
    elif args.command == "restore-shelves":
        restore_all_shelves()
    else:
        raise ValueError(f"invalid command: {args.command}")


if __name__ == "__main__":
    main()
