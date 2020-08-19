from fractions import Fraction
from itertools import chain
from typing import (Iterable,
                    Iterator,
                    Sequence,
                    Tuple)

from robust.angular import (Orientation,
                            orientation)
from robust.hints import Expansion
from robust.utils import (scale_expansion,
                          sum_expansions,
                          two_product,
                          two_two_diff)

from hypothesis_geometry.hints import (Contour,
                                       Coordinate,
                                       Domain,
                                       Point)

flatten = chain.from_iterable


def pairwise(iterable: Iterable[Domain]) -> Iterable[Tuple[Domain, Domain]]:
    iterator = iter(iterable)
    element = next(iterator)
    for next_element in iterator:
        yield element, next_element
        element = next_element


Orientation = Orientation
orientation = orientation


def to_orientations(contour: Contour) -> Iterator[Orientation]:
    return (orientation(contour[index - 1], contour[index],
                        contour[(index + 1) % len(contour)])
            for index in range(len(contour)))


def points_to_centroid(points: Sequence[Point]) -> Point:
    accumulated_x = accumulated_y = 0
    for x, y in points:
        accumulated_x += x
        accumulated_y += y
    divisor = len(points)
    return (_divide_by_int(accumulated_x, divisor),
            _divide_by_int(accumulated_y, divisor))


def contour_to_centroid(contour: Contour) -> Point:
    double_area = x_numerator = y_numerator = (0,)
    prev_x, prev_y = contour[-1]
    for x, y in contour:
        area_component = _to_endpoints_cross_product_z(prev_x, prev_y, x, y)
        double_area = sum_expansions(double_area, area_component)
        x_numerator = sum_expansions(x_numerator,
                                     scale_expansion(area_component,
                                                     prev_x + x))
        y_numerator = sum_expansions(y_numerator,
                                     scale_expansion(area_component,
                                                     prev_y + y))
        prev_x, prev_y = x, y
    divisor = 3 * double_area[-1]
    return (_divide_by_int(x_numerator[-1], divisor),
            _divide_by_int(y_numerator[-1], divisor))


def _to_endpoints_cross_product_z(start_x: Coordinate,
                                  start_y: Coordinate,
                                  end_x: Coordinate,
                                  end_y: Coordinate) -> Expansion:
    minuend, minuend_tail = two_product(start_x, end_y)
    subtrahend, subtrahend_tail = two_product(start_y, end_x)
    return (two_two_diff(minuend, minuend_tail, subtrahend, subtrahend_tail)
            if minuend_tail or subtrahend_tail
            else (minuend - subtrahend,))


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
