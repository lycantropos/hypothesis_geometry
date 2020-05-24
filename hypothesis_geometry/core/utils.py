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


def contour_to_centroid(contour: Contour) -> Point:
    double_area, x_numerator, y_numerator = 0, 0, 0
    prev_x, prev_y = contour[-1]
    for x, y in contour:
        area_component = prev_x * y - prev_y * x
        double_area += area_component
        x_numerator += (prev_x + x) * area_component
        y_numerator += (prev_y + y) * area_component
        prev_x, prev_y = x, y
    denominator = 3 * double_area
    assert denominator != 0
    return (_divide_by_int(x_numerator, denominator),
            _divide_by_int(y_numerator, denominator))


def _divide_by_int(dividend: Coordinate, divisor: int) -> Coordinate:
    return (Fraction(dividend, divisor)
            if isinstance(dividend, int)
            else dividend / divisor)


def point_in_angle(point: Point,
                   first_ray_point: Point,
                   vertex: Point,
                   second_ray_point: Point) -> bool:
    angle_orientation = orientation(first_ray_point, vertex, second_ray_point)
    first_half_orientation = orientation(first_ray_point, vertex, point)
    second_half_orientation = orientation(vertex, second_ray_point, point)
    return (second_half_orientation is angle_orientation
            if first_half_orientation is Orientation.COLLINEAR
            else (first_half_orientation is angle_orientation
                  if second_half_orientation is Orientation.COLLINEAR
                  else (first_half_orientation is second_half_orientation
                        is (angle_orientation
                            # if angle is degenerate
                            or Orientation.COUNTERCLOCKWISE))))