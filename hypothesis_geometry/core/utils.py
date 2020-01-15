from enum import (IntEnum,
                  unique)
from itertools import chain
from numbers import Real
from typing import (Callable,
                    Hashable,
                    Iterable,
                    Iterator,
                    List,
                    Sequence,
                    TypeVar)

from robust import parallelogram
from robust.hints import Point as RealPoint

from hypothesis_geometry.hints import (Contour,
                                       Coordinate,
                                       Point)

Domain = TypeVar('Domain')


def to_sign(value: Real) -> int:
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0


flatten = chain.from_iterable


def unique_everseen(iterable: Iterable[Domain],
                    *,
                    key: Callable[[Domain], Hashable] = None
                    ) -> Iterable[Domain]:
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in iterable:
            if element not in seen:
                seen_add(element)
                yield element
    else:
        for element in iterable:
            value = key(element)
            if value not in seen:
                seen_add(value)
                yield element


def split(sequence: Sequence[Domain],
          *,
          size: int = 2) -> List[Sequence[Domain]]:
    step, offset = divmod(len(sequence), size)
    return [sequence[number * step + min(number, offset):
                     (number + 1) * step + min(number + 1, offset)]
            for number in range(size)]


@unique
class Orientation(IntEnum):
    CLOCKWISE = -1
    COLLINEAR = 0
    COUNTERCLOCKWISE = 1


def to_orientation(first_ray_point: Point,
                   vertex: Point,
                   second_ray_point: Point) -> Orientation:
    return Orientation(to_sign(parallelogram.signed_area(
            to_real_point(vertex), to_real_point(first_ray_point),
            to_real_point(vertex), to_real_point(second_ray_point))))


def to_orientations(contour: Contour) -> Iterator[Orientation]:
    return (to_orientation(contour[index - 1], contour[index],
                           contour[(index + 1) % len(contour)])
            for index in range(len(contour)))


def to_real_point(point: Point) -> RealPoint:
    x, y = point
    return _coordinate_to_real(x), _coordinate_to_real(y)


def _coordinate_to_real(coordinate: Coordinate) -> Real:
    return coordinate if isinstance(coordinate, Real) else float(coordinate)
