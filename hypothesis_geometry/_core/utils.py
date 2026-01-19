from collections.abc import Callable, Iterable, Sequence
from functools import partial
from itertools import chain
from typing import Any, Protocol, TypeVar

from bentley_ottmann.planar import segments_cross_or_overlap
from ground.hints import Contour, Segment
from typing_extensions import Self

from .hints import Domain, Range, ScalarT


def apply(function: Callable[..., Range], args: Iterable[Domain]) -> Range:
    return function(*args)


def ceil_log2(number: int) -> int:
    return number.bit_length() - (not (number & (number - 1)))


def cut(values: Sequence[Domain], limit: int) -> Sequence[Domain]:
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


def pack(
    function: Callable[..., Range], /
) -> Callable[[Iterable[Any]], Range]:
    return partial(apply, function)


def pairwise(iterable: Iterable[Domain], /) -> Iterable[tuple[Domain, Domain]]:
    iterator = iter(iterable)
    try:
        element = next(iterator)
    except StopIteration:
        return
    for next_element in iterator:
        yield element, next_element
        element = next_element


class Ordered(Protocol):
    def __lt__(self, other: Self, /) -> bool: ...


_OrderedT = TypeVar('_OrderedT', bound=Ordered)


def sort_pair(
    pair: Sequence[_OrderedT], /, *, reverse: bool = False
) -> tuple[_OrderedT, _OrderedT]:
    first, second = pair
    return (first, second) if first < second or reverse else (second, first)


def contours_do_not_cross_or_overlap(
    contours: Sequence[Contour[ScalarT]],
    contour_segments_factory: Callable[
        [Contour[ScalarT]], Sequence[Segment[ScalarT]]
    ],
    /,
) -> bool:
    return segments_do_not_cross_or_overlap(
        list(
            flatten(contour_segments_factory(contour) for contour in contours)
        )
    )


def segments_do_not_cross_or_overlap(
    segments: Sequence[Segment[ScalarT]],
) -> bool:
    return not segments_cross_or_overlap(segments)
