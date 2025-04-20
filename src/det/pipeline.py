from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import Any, Unpack


class Pipe[T]:
    def __init__(self, arg: T) -> None:
        self._value: T = arg

    @property
    def value(self) -> T:
        return self._value

    def __rshift__[R](self, other: Callable[[T], R]) -> Pipe[R]:
        return Pipe(other(self._value))


def identity[T](arg: T) -> T:
    return arg


def reverse_composition[X, Y, Z](
    func1: Callable[[X], Y], func2: Callable[[Y], Z]
) -> Callable[[X], Z]:
    return lambda x: func2(func1(x))


type Fns[T, R] = tuple[
    Callable[[T], Any], Unpack[tuple[Callable[[Any], Any], ...]], Callable[[Any], R]  # pyright: ignore[reportExplicitAny]
]


class Pipeline[T, R]:
    def __init__(self, *fns: Unpack[Fns[T, R]]) -> None:
        self.func: Callable[[T], R] = reduce(reverse_composition, fns)

    def __getitem__(self, arg: T) -> R:
        return self.func(arg)
