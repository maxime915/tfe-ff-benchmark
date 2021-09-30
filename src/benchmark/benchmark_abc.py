"benchmark_abc : base class for defining and running benchmarks"

# python v3.{7, 8, 9} needs this for typing.Type[Benchmark] to work
# see https://stackoverflow.com/questions/45636816/typing-static-methods-returning-class-instance
from __future__ import annotations

import abc
import collections
import enum
import timeit
import typing

import numpy


class Verbosity(enum.Enum):
    "verbosity level, prefer 2 for debug and 0 for piping output"
    QUIET = 0
    DEFAULT = 1
    VERBOSE = 2


class BenchmarkABC(abc.ABC):
    "benchmark object: defining info & run"

    # dictionary to map file extension to classes -> acts as a factory
    _mapper: typing.Dict[str, typing.Type[BenchmarkABC]] = {}

    @classmethod
    def __init_subclass__(cls, /, re_str: str = '', **kwargs):
        """Register all subclasses for easy regex -> Class mapping. (opt)
        re_str must be a valid regex against which filename will be checked

        To avoid registration, use re_str='' or any falsy value."""

        super().__init_subclass__(**kwargs)

        if not re_str:
            return

        if re_str in BenchmarkABC._mapper:
            raise ValueError((f'cannot use {re_str} for {cls.__name__} : it is '
                              f'used for {BenchmarkABC._mapper[re_str].__name__}'))
        BenchmarkABC._mapper[re_str] = cls

    @classmethod
    def get_ordered_mapper(cls) -> typing.Dict[str, typing.Type[BenchmarkABC]]:
        "returns a dictionary with longest prefix first"

        def key_len(entry: typing.Tuple[str, typing.Type[BenchmarkABC]]):
            return len(entry[0])
        return collections.OrderedDict(sorted(cls._mapper.items(), key=key_len, reverse=True))

    @abc.abstractmethod
    def __init__(self, path: str, *args, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def info(self, verbosity: Verbosity) -> str:
        "give information on the benchmark and the file data (chunk, size, ...)"

    @abc.abstractmethod
    def task(self) -> None:
        "intensive task of the benchmark to be repeated"

    def bench(self, verbosity: Verbosity = Verbosity.DEFAULT, number: int = 1_000,
              repeat: int = 3, enable_gc: bool = False) -> None:
        "perform the benchmark with given parameters"

        setup = 'pass'
        if enable_gc:
            setup = 'import gc\ngc.enable()'

        timer = timeit.Timer(self.task, setup=setup)

        results = numpy.array(timer.repeat(repeat, number))
        results /= number

        if verbosity == Verbosity.QUIET:
            print(results.min())
            return

        print((f'{self.__class__.__name__} (duration averaged on {number} '
               f'iteration{"s" if number > 0 else ""}, repeated {repeat} '
               f'time{"s" if repeat > 0 else ""}.)'))

        if verbosity == Verbosity.DEFAULT:
            print(f'{results.min():5.3e} s')
        elif verbosity == Verbosity.VERBOSE:
            res = "[" + ", ".join([f'{v:5.3e}' for v in results]) + "]"
            print((f'results: {res}: {results.min():5.3e} s to {results.max():5.3e} s, '
                   f'{results.mean():5.3e} s ± {results.std(ddof=1):5.3e} s'))

        info = self.info(verbosity)
        if info:
            print(info)
        
        # final newline for clarity
        print('')
