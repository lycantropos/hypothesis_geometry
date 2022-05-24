from functools import partial
from itertools import chain
from typing import (Callable,
                    Iterable,
                    Sequence,
                    Tuple)

from .hints import (Domain,
                    Range)


def apply(function: Callable[..., Range],
          args: Iterable[Domain]) -> Range:
    return function(*args)


def ceil_log2(number: int) -> int:
    return number.bit_length() - (not (number & (number - 1)))


def cut(values: Domain, limit: int) -> Domain:
    return values[:limit] if limit < len(values) else values


flatten = chain.from_iterable


def to_prior_prime(value: int) -> int:
    assert value > 2, value
    step = value + ((value & 1) - 1)
    while not _is_prime(step):
        step -= 2
    return step


def to_next_prime(value: int) -> int:
    assert value > 2, value
    step = value | 1
    while not _is_prime(step):
        step += 2
    return step


def _is_prime(value: int) -> bool:
    assert value % 2 != 0
    if value % 3 == 0:
        return False
    divisor = 6
    while divisor * divisor - 2 * divisor + 1 <= value:
        if value % (divisor - 1) == 0:
            return False
        if value % (divisor + 1) == 0:
            return False
        divisor += 6
    return True


def pack(function: Callable[..., Range]
         ) -> Callable[[Iterable[Domain]], Range]:
    return partial(apply, function)


def pairwise(iterable: Iterable[Domain]) -> Iterable[Tuple[Domain, Domain]]:
    iterator = iter(iterable)
    element = next(iterator, None)
    for next_element in iterator:
        yield element, next_element
        element = next_element


def sort_pair(pair: Sequence[Domain]) -> Tuple[Domain, Domain]:
    first, second = pair
    return (first, second) if first < second else (second, first)
