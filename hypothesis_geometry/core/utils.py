from functools import partial
from itertools import chain
from typing import (Callable,
                    Iterable,
                    Sequence,
                    Tuple)

from .hints import (Domain,
                    Range)

flatten = chain.from_iterable


def pairwise(iterable: Iterable[Domain]) -> Iterable[Tuple[Domain, Domain]]:
    iterator = iter(iterable)
    element = next(iterator, None)
    for next_element in iterator:
        yield element, next_element
        element = next_element


def apply(function: Callable[..., Range],
          args: Iterable[Domain]) -> Range:
    return function(*args)


def cut(values: Domain, limit: int) -> Domain:
    return values[:limit] if limit < len(values) else values


def pack(function: Callable[..., Range]
         ) -> Callable[[Iterable[Domain]], Range]:
    return partial(apply, function)


def sort_pair(pair: Sequence[Domain]) -> Tuple[Domain, Domain]:
    first, second = pair
    return (first, second) if first < second else (second, first)


