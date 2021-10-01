"utils: utility module for benchmarks"

import argparse
import typing


def _gt0_int(value) -> int:
    "parser positive int"
    try:
        as_int = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value} is not a positive integer')

    if as_int <= 0:
        raise argparse.ArgumentTypeError(f'{value} is not a positive integer')
    return as_int


def _arg_range(min: int, max: int) -> typing.Type[argparse.Action]:
    """parse ranged number of args, see
    https://stackoverflow.com/questions/4194948/\
python-argparse-is-there-a-way-to-specify-a-range-in-nargs"""
    class _ArgRange(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if len(values) > max or len(values) < min:
                raise argparse.ArgumentTypeError((f'argument "{self.dest}" requires'
                                                  f' between {min} and {max} arguments.'))
            setattr(args, self.dest, values)
    return _ArgRange


def basic_parser() -> argparse.ArgumentParser:
    "return an ArgumentParser suitable for gc, number, repeat and files"

    parser = argparse.ArgumentParser()

    gc_group = parser.add_mutually_exclusive_group()

    gc_group.add_argument('--no-gc', dest='gc', action='store_false',
                          help="disable the garbage collector during measurements (default)")
    gc_group.add_argument('--gc', dest='gc', action='store_true',
                          help="enable the garbage collector during measurements")
    gc_group.set_defaults(gc=False)

    parser.add_argument('--number', type=int, default=1_000, help='see timeit')
    parser.add_argument('--repeat', type=int, default=3, help='see timeit')

    parser.add_argument('files', nargs='+', type=str,
                        help='files to benchmark')

    return parser


def tiled_parser(ndim: int = 2, default: typing.Iterable[int] = (32,)) -> argparse.ArgumentParser:
    "return a basic parser augmented with tile support"
    parser = basic_parser()

    parser.add_argument('--tile', nargs='+', default=list(default),
                        action=_arg_range(1, ndim), type=_gt0_int,
                        help=f'tile size, at least 1, max {ndim}')

    return parser


def parse_args() -> argparse.Namespace:
    "parse the arguments of a basic parser (see basic_parser)"
    return basic_parser().parse_args()


def parse_args_with_tile(ndim: int = 2, default: typing.Iterable[int] = (32,)) -> argparse.Namespace:
    "parse the arguments of a tiled parser (see tiled_parser)"
    return tiled_parser(ndim=ndim, default=default).parse_args()
