import warnings
from functools import partial
from itertools import repeat
from typing import (Optional,
                    Sequence,
                    Tuple)

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
from .utils import (to_concave_contour,
                    to_convex_contour,
                    to_convex_hull)

TRIANGLE_SIZE = 3
MIN_CONCAVE_CONTOUR_SIZE = 4


def points(x_coordinates: Strategy[Coordinate],
           y_coordinates: Optional[Strategy[Coordinate]] = None,
           ) -> Strategy[Point]:
    """
    Returns a strategy for points.

    :param x_coordinates: strategy for points' x-coordinates.
    :param y_coordinates:
        strategy for points' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    """
    if y_coordinates is None:
        y_coordinates = x_coordinates
    return strategies.tuples(x_coordinates, y_coordinates)


def contours(x_coordinates: Strategy[Coordinate],
             y_coordinates: Optional[Strategy[Coordinate]] = None,
             *,
             min_size: int = TRIANGLE_SIZE,
             max_size: Optional[int] = None) -> Strategy[Contour]:
    """
    Returns a strategy for contours.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for contour size.
    :param max_size: upper bound for contour size, ``None`` for unbound.
    """
    _validate_sizes(min_size, max_size)
    if max_size is not None and max_size == TRIANGLE_SIZE:
        return triangular_contours(x_coordinates, y_coordinates)
    return (convex_contours(x_coordinates, y_coordinates,
                            min_size=min_size,
                            max_size=max_size)
            | concave_contours(x_coordinates, y_coordinates,
                               min_size=min_size,
                               max_size=max_size))


def concave_contours(x_coordinates: Strategy[Coordinate],
                     y_coordinates: Optional[Strategy[Coordinate]] = None,
                     *,
                     min_size: int = MIN_CONCAVE_CONTOUR_SIZE,
                     max_size: Optional[int] = None) -> Strategy[Contour]:
    """
    Returns a strategy for concave contours.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for contour size.
    :param max_size: upper bound for contour size, ``None`` for unbound.
    """
    _validate_sizes(min_size, max_size, MIN_CONCAVE_CONTOUR_SIZE)
    return (strategies.lists(points(x_coordinates, y_coordinates),
                             min_size=min_size,
                             max_size=max_size,
                             unique=True)
            .filter(points_do_not_lie_on_the_same_line)
            .map(triangular.delaunay)
            .map(to_concave_contour)
            .filter(partial(_contour_has_valid_size,
                            min_size=min_size,
                            max_size=max_size))
            .filter(is_contour_non_convex)
            .filter(is_non_self_intersecting_contour))


def convex_contours(x_coordinates: Strategy[Coordinate],
                    y_coordinates: Optional[Strategy[Coordinate]] = None,
                    *,
                    min_size: int = TRIANGLE_SIZE,
                    max_size: Optional[int] = None) -> Strategy[Contour]:
    """
    Returns a strategy for convex contours.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for contour size.
    :param max_size: upper bound for contour size, ``None`` for unbound.
    """
    _validate_sizes(min_size, max_size)
    if max_size is not None and max_size == TRIANGLE_SIZE:
        return triangular_contours(x_coordinates, y_coordinates)
    min_size = max(min_size, TRIANGLE_SIZE)

    def to_coordinates_with_flags_and_permutations(
            coordinates_with_flags: Sequence[Tuple[Coordinate, Coordinate,
                                                   bool, bool]]
    ) -> Strategy[Tuple[Sequence[Tuple[Point, bool, bool]], Sequence[int]]]:
        indices = range(len(coordinates_with_flags))
        return strategies.tuples(strategies.just(coordinates_with_flags),
                                 strategies.permutations(indices))

    result = (strategies.lists(strategies.tuples(x_coordinates,
                                                 x_coordinates
                                                 if y_coordinates is None
                                                 else y_coordinates,
                                                 strategies.booleans(),
                                                 strategies.booleans()),
                               min_size=min_size,
                               max_size=max_size,
                               unique=True)
              .flatmap(to_coordinates_with_flags_and_permutations)
              .map(to_convex_contour)
              .map(to_convex_hull)
              .filter(partial(_contour_has_valid_size,
                              min_size=min_size,
                              max_size=max_size)))
    return (triangular_contours(x_coordinates, y_coordinates) | result
            if min_size <= TRIANGLE_SIZE
            else result)


def triangular_contours(x_coordinates: Strategy[Coordinate],
                        y_coordinates: Optional[Strategy[Coordinate]] = None,
                        ) -> Strategy[Contour]:
    """
    Returns a strategy for triangular contours.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' x-coordinates,
        ``None`` for reusing x-coordinates strategy.
    """
    return (strategies.tuples(*repeat(points(x_coordinates, y_coordinates),
                                      times=3))
            .filter(is_contour_strict)
            .map(list))


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
