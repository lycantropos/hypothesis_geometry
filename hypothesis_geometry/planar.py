from itertools import repeat
from typing import Optional

from hypothesis import strategies

from .core import triangular
from .core.contracts import (is_contour_non_convex,
                             is_contour_strict,
                             is_non_self_intersecting_contour,
                             points_do_not_lie_on_the_same_line)
from .hints import (Contour,
                    Point,
                    Scalar,
                    Strategy)
from .utils import (to_convex_hull,
                    triangulation_to_concave_contour)

TRIANGLE_SIZE = 3


def points(coordinates: Strategy[Scalar]) -> Strategy[Point]:
    return strategies.tuples(coordinates, coordinates)


def contours(coordinates: Strategy[Scalar],
             *,
             min_size: int = TRIANGLE_SIZE,
             max_size: Optional[int] = None) -> Strategy[Contour]:
    _validate_sizes(min_size, max_size)
    if max_size is not None and max_size == TRIANGLE_SIZE:
        return triangles(coordinates)
    return convex_contours(coordinates) | concave_contours(coordinates)


def _validate_sizes(min_size: int, max_size: Optional[int]) -> None:
    if min_size < TRIANGLE_SIZE:
        raise ValueError('Contours should have at least {expected} vertices, '
                         'but requested {actual}.'
                         .format(expected=TRIANGLE_SIZE,
                                 actual=min_size))
    if max_size is not None and min_size > max_size:
        raise ValueError('`min_size` should not be greater than `max_size`,'
                         'but found {min_size}, {max_size}.'
                         .format(min_size=min_size,
                                 max_size=max_size))


def convex_contours(coordinates: Strategy[Scalar]) -> Strategy[Contour]:
    return (triangles(coordinates)
            | (strategies.lists(points(coordinates),
                                min_size=4,
                                unique=True)
               .map(to_convex_hull)
               .filter(lambda contour: len(contour) >= 3)))


def triangles(coordinates: Strategy[Scalar]) -> Strategy[Contour]:
    return (strategies.tuples(*repeat(points(coordinates),
                                      times=3))
            .filter(is_contour_strict)
            .map(list))


def concave_contours(coordinates: Strategy[Scalar]) -> Strategy[Contour]:
    return (strategies.lists(points(coordinates),
                             min_size=4,
                             unique=True)
            .filter(points_do_not_lie_on_the_same_line)
            .map(triangular.delaunay)
            .map(triangulation_to_concave_contour)
            .filter(lambda contour: len(contour) > 2)
            .filter(is_contour_non_convex)
            .filter(is_non_self_intersecting_contour))
