"present: script to show a graph of the different access types"

import argparse
import enum
import sys
from collections import OrderedDict
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .inspect_db import DB


_DEFAULT_FIGSIZE = (6, 3.5)


class Command(enum.Enum):
    BAND = enum.auto()
    TIC = enum.auto()
    SEARCH = enum.auto()
    OVERLAP = enum.auto()
    GET_CHUNKS = enum.auto()

    @staticmethod
    def str_choices() -> List[str]:
        return [choice.name.lower() for choice in Command]

    @staticmethod
    def parse(text: str) -> "Command":
        mapper = {choice.name.lower(): choice for choice in Command}

        if text.lower() in mapper:
            return mapper[text.lower()]

        raise ValueError(f"invalid Command: {text}")


if __name__ != "__main__":
    sys.exit("should not be imported")

COLORS = list("bgrcmyk")

parser = argparse.ArgumentParser()
parser.add_argument(
    "command", choices=["all"] + Command.str_choices(), help="command to run"
)
parser.add_argument("file", type=str, help="shelf store file")

args = parser.parse_args()

data = OrderedDict(sorted(DB(args.file).load_all().items(), key=str))

if args.command == "all":
    PARSED_COMMANDS = list(Command)
else:
    PARSED_COMMANDS = [Command.parse(args.command)]


def run_on(command: Command):
    "runs the annotated function if command is in PARSED_COMMANDS"

    def _runner(function: "function"):
        if command in PARSED_COMMANDS:
            print(f"running {function.__name__=} for {command.name=}")
            function()
        return function
    return _runner


@run_on(Command.BAND)
def present_band():

    time_dict = {k: v for k, v in data.items() if "band" in k}

    zarr_number = data[("benchmark infos",)]["zarr number"]

    mode = "continuous" if data[("imzml info",)].continuous_mode else "processed"

    fig = plt.figure(figsize=_DEFAULT_FIGSIZE)
    axis = fig.add_subplot(111)

    axis.set_yscale("log")

    # load zarr data (one for each chunk)
    zarr_dict = {k: v for k, v in time_dict.items() if "imzml_zarr" in k}
    if not zarr_dict:
        raise ValueError("no band found")

    zarr_arrays = [np.array(v) / zarr_number for v in zarr_dict.values()]

    means = [z.mean() for z in zarr_arrays]
    yerrs = [z.std() for z in zarr_arrays]

    keys = [k[1] for k in zarr_dict.keys()]

    axis.plot(keys, means, "--x")
    axis.fill_between(
        keys,
        [m - y for m, y in zip(means, yerrs)],
        [m + y for m, y in zip(means, yerrs)],
        alpha=0.3,
    )

    limit_high = axis.get_ylim()[1]
    for i, val in enumerate(means):
        if val > limit_high:
            axis.text(i - 0.25, limit_high - 0.5, f"{val:.2f}")

    axis.grid(axis='y', alpha=0.7)
    
    axis.set_title(f"Spectral Access time (single band) - {mode} mode file")
    axis.set_ylabel("Mean time (s)")

    fig.tight_layout()
    fig.savefig(f"band_{mode}.png")


@run_on(Command.TIC)
def present_tic():
    time_dict = {k: v for k, v in data.items() if "tic" in k}

    zarr_number = data[("benchmark infos",)]["zarr number"]
    windows = data[("benchmark parameters",)]["tile_choice"]

    mode = "continuous" if data[("imzml info",)].continuous_mode else "processed"

    fig = plt.figure(figsize=_DEFAULT_FIGSIZE)
    axis = fig.add_subplot(111)

    axis.set_yscale("log")

    for window_tpl, color in zip(windows, COLORS):
        # load zarr data (one for each chunk)
        zarr_dict = {
            k: v for k, v in time_dict.items() if window_tpl in k and "imzml_zarr" in k
        }
        if not zarr_dict:
            print(f"skipping {window_tpl=}")
            continue

        zarr_arrays = [np.array(v) / zarr_number for v in zarr_dict.values()]

        means = [z.mean() for z in zarr_arrays]
        yerrs = [z.std() for z in zarr_arrays]

        keys = [k[1] for k in zarr_dict.keys()]

        axis.plot(keys, means, "--x", color=color, label=str(window_tpl))
        axis.fill_between(
            keys,
            [m - y for m, y in zip(means, yerrs)],
            [m + y for m, y in zip(means, yerrs)],
            color=color,
            alpha=0.3,
        )

        limit_high = axis.get_ylim()[1]
        for i, val in enumerate(means):
            if val > limit_high:
                axis.text(i - 0.25, limit_high - 0.5, f"{val:.2f}")

    axis.grid(axis='y', alpha=0.7)
    
    axis.set_title(f"Channel Sum time (sum over all bands) - {mode} mode file")
    axis.set_ylabel("Mean time (s)")
    axis.legend()

    fig.tight_layout()
    fig.savefig(f"tic_{mode}.png")


@run_on(Command.SEARCH)
def present_search():
    time_dict = {k: v for k, v in data.items() if "search" in k}

    zarr_number = data[("benchmark infos",)]["zarr number"]
    windows = data[("benchmark parameters",)]["tile_choice"]

    mode = "continuous" if data[("imzml info",)].continuous_mode else "processed"

    fig = plt.figure(figsize=_DEFAULT_FIGSIZE)
    axis = fig.add_subplot(111)

    axis.set_yscale("log")

    for window_tpl, color in zip(windows, COLORS):
        # load zarr data (one for each chunk)
        zarr_dict = {
            k: v for k, v in time_dict.items() if window_tpl in k and "imzml_zarr" in k
        }
        if not zarr_dict:
            print(f"skipping {window_tpl=}")
            continue

        zarr_arrays = [np.array(v) / zarr_number for v in zarr_dict.values()]

        means = [z.mean() for z in zarr_arrays]
        yerrs = [z.std() for z in zarr_arrays]

        keys = [k[1] for k in zarr_dict.keys()]

        axis.plot(keys, means, "--x", color=color, label=str(window_tpl))
        axis.fill_between(
            keys,
            [m - e for m, e in zip(means, yerrs)],
            [m + e for m, e in zip(means, yerrs)],
            color=color,
            alpha=0.3,
        )

        limit_high = axis.get_ylim()[1]
        for i, val in enumerate(means):
            if val > limit_high:
                axis.text(i - 0.25, limit_high - 0.5, f"{val:.2f}")
    
    axis.grid(axis='y', alpha=0.7)

    axis.set_title(f"Channel Search time - {mode} mode file")
    axis.set_ylabel("Mean time (s)")
    axis.legend()

    fig.tight_layout()
    fig.savefig(f"search_{mode}.png")


@run_on(Command.OVERLAP)
def present_overlap():

    time_dict = {k: v for k, v in data.items() if "tic-overlap" in k}
    
    chunk_choices = data[("benchmark parameters",)]["chunk_choice"]
    chunk_as_tpl = {}

    for chunk in chunk_choices:
        infos = dict(data[("imzml_zarr", chunk, "infos", "intensities")])
        chunk_as_tpl[chunk] = infos["Chunk shape"]

    windows = data[("benchmark parameters",)]["tile_choice"]

    mode = "continuous" if data[("imzml info",)].continuous_mode else "processed"

    fig = plt.figure(figsize=_DEFAULT_FIGSIZE)
    axis = fig.add_subplot(111)

    # axis.set_ylim((1.6, 2.1))
    axis.set_ylim((2.8, 4.2))

    for window_tpl, color in zip(windows, COLORS):

        # load zarr data (one for each chunk)
        overlap_00_dict = {
            k[1]: v
            for k, v in time_dict.items()
            if k[-3:] == ("tic-overlap", window_tpl, (0, 0))
        }
        overlap_11_dict = {
            k[1]: v
            for k, v in time_dict.items()
            if k[-3:] == ("tic-overlap", window_tpl, (1, 1))
        }

        if not overlap_00_dict:
            print(f"skipping {window_tpl=}")
            continue

        ratios = {}
        for chunk in overlap_00_dict.keys() & overlap_11_dict.keys():
            o00 = overlap_00_dict[chunk]
            o11 = overlap_11_dict[chunk]

            ratios[chunk] = np.mean(o11) / np.mean(o00)
        
        sorted_ratios = OrderedDict()
        for key in sorted(chunk_as_tpl.keys()):
            sorted_ratios[key] = ratios.get(key, np.nan)

        axis.scatter(
            sorted_ratios.keys(),
            sorted_ratios.values(),
            c=color,
            label=str(window_tpl)
        )
        
    axis.grid(axis='y', alpha=0.7)

    axis.set_title(f"Overlap time ratio on a window of size - {mode} mode file")
    axis.set_ylabel("Time ratio")
    axis.legend()

    fig.tight_layout()
    fig.savefig(f"ticoverlap_{mode}_00_11.png")


@run_on(Command.GET_CHUNKS)
def present_get_chunks():

    chunk_choices = data[("benchmark parameters",)]["chunk_choice"]
    chunk_as_tpl = {}

    for chunk in chunk_choices:
        infos = dict(data[("imzml_zarr", chunk, "infos", "intensities")])
        chunk_as_tpl[chunk] = infos["Chunk shape"]

    for chunk, tpl in chunk_as_tpl.items():

        print(f"{chunk} : {tpl} -> {np.prod(eval(tpl)):.2E} elements per chunks")
