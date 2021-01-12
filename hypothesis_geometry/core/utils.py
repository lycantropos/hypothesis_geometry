from functools import partial
from itertools import chain
from typing import (Callable,
                    Iterable,
                    Sequence,
                    Tuple)

from ground.base import (Context,
                         Orientation)
from ground.hints import Point

from .hints import (Domain,
                    Orienteer,
                    Range)

flatten = chain.from_iterable


def pairwise(iterable: Iterable[Domain]) -> Iterable[Tuple[Domain, Domain]]:
    iterator = iter(iterable)
    element = next(iterator, None)
    for next_element in iterator:
        yield element, next_element
        element = next_element


def to_contour_orienteer(context: Context) -> Callable[[Sequence[Point]],
                                                       Iterable[Orientation]]:
    return partial(_to_contour_orientations, context.angle_orientation)


def _to_contour_orientations(orienteer: Orienteer,
                             vertices: Sequence[Point]
                             ) -> Iterable[Orientation]:
    return (orienteer(vertices[index], vertices[index - 1],
                      vertices[(index + 1) % len(vertices)])
            for index in range(len(vertices)))


def apply(function: Callable[..., Range],
          args: Iterable[Domain]) -> Range:
    return function(*args)


def pack(function: Callable[..., Range]
         ) -> Callable[[Iterable[Domain]], Range]:
    return partial(apply, function)


def sort_pair(pair: Sequence[Domain]) -> Tuple[Domain, Domain]:
    first, second = pair
    return (first, second) if first < second else (second, first)
