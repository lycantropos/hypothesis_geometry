import warnings
from functools import partial
from itertools import repeat
from operator import itemgetter
from typing import Optional

from hypothesis import strategies
from hypothesis.errors import HypothesisWarning

from .core import triangular
from .core.contracts import (is_contour_non_convex,
                             is_contour_strict,
                             is_non_self_intersecting_contour,
                             points_do_not_lie_on_the_same_line)
from .hints import (Contour,
                    Coordinate,
                    Point,
                    Strategy)
from .utils import (to_convex_hull,
                    triangulation_to_concave_contour)

TRIANGLE_SIZE = 3
MIN_CONCAVE_CONTOUR_SIZE = 4


def points(coordinates: Strategy[Coordinate]) -> Strategy[Point]:
    return strategies.tuples(coordinates, coordinates)


def contours(coordinates: Strategy[Coordinate],
             *,
             min_size: int = TRIANGLE_SIZE,
             max_size: Optional[int] = None) -> Strategy[Contour]:
    _validate_sizes(min_size, max_size)
    if max_size is not None and max_size == TRIANGLE_SIZE:
        return triangular_contours(coordinates)
    return (convex_contours(coordinates,
                            min_size=min_size,
                            max_size=max_size)
            | concave_contours(coordinates,
                               min_size=min_size,
                               max_size=max_size))


def convex_contours(coordinates: Strategy[Coordinate],
                    *,
                    min_size: int = TRIANGLE_SIZE,
                    max_size: Optional[int] = None) -> Strategy[Contour]:
    _validate_sizes(min_size, max_size)
    if max_size is not None and max_size == TRIANGLE_SIZE:
        return triangular_contours(coordinates)
    min_size = max(min_size, TRIANGLE_SIZE)
    result = (strategies.lists(points(coordinates),
                               min_size=min_size,
                               unique=True)
              .map(to_convex_hull)
              .map(itemgetter(slice(0, max_size)))
              .filter(partial(_contour_has_valid_size,
                              min_size=min_size,
                              max_size=max_size))
              .filter(is_contour_strict))
    return (triangular_contours(coordinates) | result
            if min_size <= TRIANGLE_SIZE
            else result)


def triangular_contours(coordinates: Strategy[Coordinate]
                        ) -> Strategy[Contour]:
    return (strategies.tuples(*repeat(points(coordinates),
                                      times=3))
            .filter(is_contour_strict)
            .map(list))


def concave_contours(coordinates: Strategy[Coordinate],
                     *,
                     min_size: int = MIN_CONCAVE_CONTOUR_SIZE,
                     max_size: Optional[int] = None) -> Strategy[Contour]:
    _validate_sizes(min_size, max_size, MIN_CONCAVE_CONTOUR_SIZE)
    return (strategies.lists(points(coordinates),
                             min_size=min_size,
                             unique=True)
            .filter(points_do_not_lie_on_the_same_line)
            .map(triangular.delaunay)
            .map(triangulation_to_concave_contour)
            .filter(partial(_contour_has_valid_size,
                            min_size=min_size,
                            max_size=max_size))
            .filter(is_contour_non_convex)
            .filter(is_non_self_intersecting_contour))


def _validate_sizes(min_size: int, max_size: Optional[int],
                    min_expected_size: int = TRIANGLE_SIZE) -> None:
    if max_size is None:
        return
    elif max_size < min_expected_size:
        raise ValueError('Contours should have at least {expected} vertices, '
                         'but requested {actual}.'
                         .format(expected=min_expected_size,
                                 actual=max_size))
    elif min_size > max_size:
        raise ValueError('`min_size` should not be greater than `max_size`, '
                         'but found {min_size}, {max_size}.'
                         .format(min_size=min_size,
                                 max_size=max_size))
    elif min_size < min_expected_size:
        warnings.warn('`min_size` is expected to be not less than {expected}, '
                      'but found {actual}.'
                      .format(expected=min_expected_size,
                              actual=min_size),
                      HypothesisWarning)


def _contour_has_valid_size(contour: Contour,
                            *,
                            min_size: int,
                            max_size: Optional[int]) -> bool:
    size = len(contour)
    return min_size <= size and (max_size is None or size <= max_size)
