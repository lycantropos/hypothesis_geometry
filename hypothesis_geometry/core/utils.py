from fractions import Fraction
from itertools import chain
from typing import (Iterator,
                    List,
                    Sequence)

from robust.angular import (Orientation,
                            orientation)

from hypothesis_geometry.hints import (Contour,
                                       Coordinate,
                                       Domain,
                                       Point)

flatten = chain.from_iterable


def split(sequence: Sequence[Domain],
          *,
          size: int = 2) -> List[Sequence[Domain]]:
    step, offset = divmod(len(sequence), size)
    return [sequence[number * step + min(number, offset):
                     (number + 1) * step + min(number + 1, offset)]
            for number in range(size)]


Orientation = Orientation
orientation = orientation


def to_orientations(contour: Contour) -> Iterator[Orientation]:
    return (orientation(contour[index - 1], contour[index],
                        contour[(index + 1) % len(contour)])
            for index in range(len(contour)))


def points_to_centroid(points: Sequence[Point]) -> Point:
    xs, ys = zip(*points)
    return (_divide_by_int(sum(xs), len(points)),
            _divide_by_int(sum(ys), len(points)))


def _divide_by_int(dividend: Coordinate, divisor: int) -> Coordinate:
    return (Fraction(dividend, divisor)
            if isinstance(dividend, int)
            else dividend / divisor)
