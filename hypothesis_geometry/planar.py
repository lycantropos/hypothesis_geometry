import warnings
from functools import partial
from itertools import (groupby,
                       repeat)
from typing import (List,
                    Optional,
                    Sequence,
                    Sized,
                    Tuple)

from hypothesis import strategies
from hypothesis.errors import HypothesisWarning

from .core.contracts import (is_contour_non_convex,
                             is_contour_strict,
                             is_non_self_intersecting_contour,
                             points_do_not_lie_on_the_same_line)
from .hints import (Contour,
                    Coordinate,
                    Point,
                    Polyline,
                    Segment,
                    Strategy)
from .utils import (pack,
                    to_concave_contour,
                    to_convex_contour)


def points(x_coordinates: Strategy[Coordinate],
           y_coordinates: Optional[Strategy[Coordinate]] = None,
           ) -> Strategy[Point]:
    """
    Returns a strategy for points.
    Point is a pair of numbers.

    :param x_coordinates: strategy for points' x-coordinates.
    :param y_coordinates:
        strategy for points' y-coordinates,
        ``None`` for reusing x-coordinates strategy.

    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate, 
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> points = planar.points(coordinates)
    >>> point = points.example()
    >>> isinstance(point, tuple)
    True
    >>> len(point) == 2
    True
    >>> all(isinstance(coordinate, coordinates_type)
    ...     for coordinate in point)
    True
    >>> all(min_coordinate <= coordinate <= max_coordinate
    ...     for coordinate in point)
    True

    For different coordinates' domains:

    >>> min_x_coordinate, max_x_coordinate = -1., 1.
    >>> min_y_coordinate, max_y_coordinate = 10., 100.
    >>> coordinates_type = float
    >>> x_coordinates = strategies.floats(min_x_coordinate, max_x_coordinate, 
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> y_coordinates = strategies.floats(min_y_coordinate, max_y_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> points = planar.points(x_coordinates, y_coordinates)
    >>> point = points.example()
    >>> isinstance(point, tuple)
    True
    >>> len(point) == 2
    True
    >>> all(isinstance(coordinate, coordinates_type)
    ...     for coordinate in point)
    True
    >>> point_x, point_y = point
    >>> min_x_coordinate <= point_x <= max_x_coordinate
    True
    >>> min_y_coordinate <= point_y <= max_y_coordinate
    True
    """
    if y_coordinates is None:
        y_coordinates = x_coordinates
    return strategies.tuples(x_coordinates, y_coordinates)


def segments(x_coordinates: Strategy[Coordinate],
             y_coordinates: Optional[Strategy[Coordinate]] = None
             ) -> Strategy[Segment]:
    """
    Returns a strategy for segments.
    Segment is a pair of unequal points.

    :param x_coordinates: strategy for endpoints' x-coordinates.
    :param y_coordinates:
        strategy for endpoints' y-coordinates,
        ``None`` for reusing x-coordinates strategy.

    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> segments = planar.segments(coordinates)
    >>> segment = segments.example()
    >>> isinstance(segment, tuple)
    True
    >>> len(segment) == 2
    True
    >>> all(isinstance(endpoint, tuple) for endpoint in segment)
    True
    >>> all(len(endpoint) == 2 for endpoint in segment)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in endpoint)
    ...     for endpoint in segment)
    True
    >>> all(all(min_coordinate <= coordinate <= max_coordinate
    ...         for coordinate in endpoint)
    ...     for endpoint in segment)
    True

    For different coordinates' domains:

    >>> min_x_coordinate, max_x_coordinate = -1., 1.
    >>> min_y_coordinate, max_y_coordinate = 10., 100.
    >>> coordinates_type = float
    >>> x_coordinates = strategies.floats(min_x_coordinate, max_x_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> y_coordinates = strategies.floats(min_y_coordinate, max_y_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> segments = planar.segments(x_coordinates, y_coordinates)
    >>> segment = segments.example()
    >>> isinstance(segment, tuple)
    True
    >>> len(segment) == 2
    True
    >>> all(isinstance(endpoint, tuple) for endpoint in segment)
    True
    >>> all(len(endpoint) == 2 for endpoint in segment)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in endpoint)
    ...     for endpoint in segment)
    True
    >>> all(min_x_coordinate <= endpoint_x <= max_x_coordinate
    ...     and min_y_coordinate <= endpoint_y <= max_y_coordinate
    ...     for endpoint_x, endpoint_y in segment)
    True
    """

    def non_degenerate_segment(segment: Segment) -> bool:
        start, end = segment
        return start != end

    points_strategy = points(x_coordinates, y_coordinates)
    return (strategies.tuples(points_strategy, points_strategy)
            .filter(non_degenerate_segment))


MIN_POLYLINE_SIZE = 2


def polylines(x_coordinates: Strategy[Coordinate],
              y_coordinates: Optional[Strategy[Coordinate]] = None,
              *,
              min_size: int = MIN_POLYLINE_SIZE,
              max_size: Optional[int] = None) -> Strategy[Polyline]:
    """
    Returns a strategy for polylines.
    Polyline is a sequence of points (called polyline's vertices)
    such that there is no consecutive equal points.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for polyline size.
    :param max_size: upper bound for polyline size, ``None`` for unbound.

    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> polylines = planar.polylines(coordinates,
    ...                              min_size=min_size,
    ...                              max_size=max_size)
    >>> polyline = polylines.example()
    >>> isinstance(polyline, list)
    True
    >>> min_size <= len(polyline) <= max_size
    True
    >>> all(isinstance(vertex, tuple) for vertex in polyline)
    True
    >>> all(len(vertex) == 2 for vertex in polyline)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in polyline)
    True
    >>> all(all(min_coordinate <= coordinate <= max_coordinate
    ...         for coordinate in vertex)
    ...     for vertex in polyline)
    True

    For different coordinates' domains:

    >>> min_x_coordinate, max_x_coordinate = -1., 1.
    >>> min_y_coordinate, max_y_coordinate = 10., 100.
    >>> coordinates_type = float
    >>> x_coordinates = strategies.floats(min_x_coordinate, max_x_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> y_coordinates = strategies.floats(min_y_coordinate, max_y_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> polylines = planar.polylines(x_coordinates, y_coordinates,
    ...                              min_size=min_size,
    ...                              max_size=max_size)
    >>> polyline = polylines.example()
    >>> isinstance(polyline, list)
    True
    >>> min_size <= len(polyline) <= max_size
    True
    >>> all(isinstance(vertex, tuple) for vertex in polyline)
    True
    >>> all(len(vertex) == 2 for vertex in polyline)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in polyline)
    True
    >>> all(min_x_coordinate <= vertex_x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex_y <= max_y_coordinate
    ...     for vertex_x, vertex_y in polyline)
    True
    """
    _validate_sizes(min_size, max_size, MIN_POLYLINE_SIZE)
    min_size = max(min_size, MIN_POLYLINE_SIZE)
    result = _polylines(x_coordinates, y_coordinates, min_size, max_size)

    if max_size is None or max_size > MIN_POLYLINE_SIZE:
        def close_polyline(polyline: Polyline) -> Polyline:
            return polyline + [polyline[0]]

        result |= (_polylines(x_coordinates, y_coordinates,
                              # closing will add a vertex,
                              # so to stay in bounds
                              # we should decrement them
                              min_size - 1
                              if min_size > MIN_POLYLINE_SIZE
                              else min_size,
                              max_size - 1
                              if max_size is not None
                              else max_size)
                   .map(close_polyline))
    return result


def _polylines(x_coordinates: Strategy[Coordinate],
               y_coordinates: Optional[Strategy[Coordinate]],
               min_size: int,
               max_size: Optional[int]) -> Strategy[Polyline]:
    def to_unique_consecutive_vertices(polyline: Polyline) -> Polyline:
        return [point for point, _ in groupby(polyline)]

    return (strategies.lists(points(x_coordinates, y_coordinates),
                             min_size=min_size,
                             max_size=max_size)
            .map(to_unique_consecutive_vertices)
            .filter(partial(_has_valid_size,
                            min_size=min_size,
                            max_size=max_size)))


TRIANGLE_SIZE = 3
MIN_CONCAVE_CONTOUR_SIZE = 4


def contours(x_coordinates: Strategy[Coordinate],
             y_coordinates: Optional[Strategy[Coordinate]] = None,
             *,
             min_size: int = TRIANGLE_SIZE,
             max_size: Optional[int] = None) -> Strategy[Contour]:
    """
    Returns a strategy for contours.
    Contour is a sequence of points (called contour's vertices)
    such that line segments formed by pairs of consecutive points
    (including the last-first point pair)
    do not cross/overlap each other.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for contour size.
    :param max_size: upper bound for contour size, ``None`` for unbound.

    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> contours = planar.contours(coordinates,
    ...                              min_size=min_size,
    ...                              max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, tuple) for vertex in contour)
    True
    >>> all(len(vertex) == 2 for vertex in contour)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True
    >>> all(all(min_coordinate <= coordinate <= max_coordinate
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True

    For different coordinates' domains:

    >>> min_x_coordinate, max_x_coordinate = -1., 1.
    >>> min_y_coordinate, max_y_coordinate = 10., 100.
    >>> coordinates_type = float
    >>> x_coordinates = strategies.floats(min_x_coordinate, max_x_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> y_coordinates = strategies.floats(min_y_coordinate, max_y_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> contours = planar.contours(x_coordinates, y_coordinates,
    ...                            min_size=min_size,
    ...                            max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, tuple) for vertex in contour)
    True
    >>> all(len(vertex) == 2 for vertex in contour)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True
    >>> all(min_x_coordinate <= vertex_x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex_y <= max_y_coordinate
    ...     for vertex_x, vertex_y in contour)
    True
    """
    _validate_sizes(min_size, max_size, TRIANGLE_SIZE)
    if max_size is not None and max_size == TRIANGLE_SIZE:
        return triangular_contours(x_coordinates, y_coordinates)
    return (convex_contours(x_coordinates, y_coordinates,
                            min_size=min_size,
                            max_size=max_size)
            | concave_contours(x_coordinates, y_coordinates,
                               min_size=max(min_size,
                                            MIN_CONCAVE_CONTOUR_SIZE),
                               max_size=max_size))


def convex_contours(x_coordinates: Strategy[Coordinate],
                    y_coordinates: Optional[Strategy[Coordinate]] = None,
                    *,
                    min_size: int = TRIANGLE_SIZE,
                    max_size: Optional[int] = None) -> Strategy[Contour]:
    """
    Returns a strategy for convex contours.
    Convex contour is a contour such that the line segment
    formed by any two points from contour's line segments
    stays inside the region bounded by the contour.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for contour size.
    :param max_size: upper bound for contour size, ``None`` for unbound.

    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> contours = planar.convex_contours(coordinates,
    ...                                   min_size=min_size,
    ...                                   max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, tuple) for vertex in contour)
    True
    >>> all(len(vertex) == 2 for vertex in contour)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True
    >>> all(all(min_coordinate <= coordinate <= max_coordinate
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True

    For different coordinates' domains:

    >>> min_x_coordinate, max_x_coordinate = -1., 1.
    >>> min_y_coordinate, max_y_coordinate = 10., 100.
    >>> coordinates_type = float
    >>> x_coordinates = strategies.floats(min_x_coordinate, max_x_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> y_coordinates = strategies.floats(min_y_coordinate, max_y_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> contours = planar.contours(x_coordinates, y_coordinates,
    ...                            min_size=min_size,
    ...                            max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, tuple) for vertex in contour)
    True
    >>> all(len(vertex) == 2 for vertex in contour)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True
    >>> all(min_x_coordinate <= vertex_x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex_y <= max_y_coordinate
    ...     for vertex_x, vertex_y in contour)
    True
    """
    _validate_sizes(min_size, max_size, TRIANGLE_SIZE)
    if max_size is not None and max_size == TRIANGLE_SIZE:
        return triangular_contours(x_coordinates, y_coordinates)
    min_size = max(min_size, TRIANGLE_SIZE)

    def to_points_with_flags_and_permutations(
            points: List[Point]) -> Strategy[Tuple[List[Point],
                                                   List[bool], List[bool],
                                                   Sequence[int]]]:
        flags = strategies.lists(strategies.booleans(),
                                 min_size=len(points),
                                 max_size=len(points))
        return strategies.tuples(strategies.just(points),
                                 flags, flags,
                                 strategies.permutations(range(len(points))))

    result = (strategies.lists(points(x_coordinates, y_coordinates),
                               min_size=min_size,
                               max_size=max_size,
                               unique=True)
              .flatmap(to_points_with_flags_and_permutations)
              .map(pack(to_convex_contour))
              .filter(partial(_has_valid_size,
                              min_size=min_size,
                              max_size=max_size)))
    return (triangular_contours(x_coordinates, y_coordinates) | result
            if min_size <= TRIANGLE_SIZE
            else result)


def concave_contours(x_coordinates: Strategy[Coordinate],
                     y_coordinates: Optional[Strategy[Coordinate]] = None,
                     *,
                     min_size: int = MIN_CONCAVE_CONTOUR_SIZE,
                     max_size: Optional[int] = None) -> Strategy[Contour]:
    """
    Returns a strategy for concave contours.
    Concave contour is a contour that is not convex.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for contour size.
    :param max_size: upper bound for contour size, ``None`` for unbound.

    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> contours = planar.concave_contours(coordinates,
    ...                                    min_size=min_size,
    ...                                    max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, tuple) for vertex in contour)
    True
    >>> all(len(vertex) == 2 for vertex in contour)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True
    >>> all(all(min_coordinate <= coordinate <= max_coordinate
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True

    For different coordinates' domains:

    >>> min_x_coordinate, max_x_coordinate = -1., 1.
    >>> min_y_coordinate, max_y_coordinate = 10., 100.
    >>> coordinates_type = float
    >>> x_coordinates = strategies.floats(min_x_coordinate, max_x_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> y_coordinates = strategies.floats(min_y_coordinate, max_y_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> contours = planar.contours(x_coordinates, y_coordinates,
    ...                            min_size=min_size,
    ...                            max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, tuple) for vertex in contour)
    True
    >>> all(len(vertex) == 2 for vertex in contour)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True
    >>> all(min_x_coordinate <= vertex_x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex_y <= max_y_coordinate
    ...     for vertex_x, vertex_y in contour)
    True
    """
    _validate_sizes(min_size, max_size, MIN_CONCAVE_CONTOUR_SIZE)
    return (strategies.lists(points(x_coordinates, y_coordinates),
                             min_size=min_size,
                             max_size=max_size,
                             unique=True)
            .filter(points_do_not_lie_on_the_same_line)
            .map(to_concave_contour)
            .filter(partial(_has_valid_size,
                            min_size=min_size,
                            max_size=max_size))
            .filter(is_contour_non_convex)
            .filter(is_non_self_intersecting_contour))


def triangular_contours(x_coordinates: Strategy[Coordinate],
                        y_coordinates: Optional[Strategy[Coordinate]] = None,
                        ) -> Strategy[Contour]:
    """
    Returns a strategy for triangular contours.
    Triangular contour is a contour formed by 3 points.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.

    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> contours = planar.triangular_contours(coordinates)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> len(contour) == 3
    True
    >>> all(isinstance(vertex, tuple) for vertex in contour)
    True
    >>> all(len(vertex) == 2 for vertex in contour)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True
    >>> all(all(min_coordinate <= coordinate <= max_coordinate
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True

    For different coordinates' domains:

    >>> min_x_coordinate, max_x_coordinate = -1., 1.
    >>> min_y_coordinate, max_y_coordinate = 10., 100.
    >>> coordinates_type = float
    >>> x_coordinates = strategies.floats(min_x_coordinate, max_x_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> y_coordinates = strategies.floats(min_y_coordinate, max_y_coordinate,
    ...                                   allow_infinity=False,
    ...                                   allow_nan=False)
    >>> contours = planar.triangular_contours(x_coordinates, y_coordinates)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> len(contour) == 3
    True
    >>> all(isinstance(vertex, tuple) for vertex in contour)
    True
    >>> all(len(vertex) == 2 for vertex in contour)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in contour)
    True
    >>> all(min_x_coordinate <= vertex_x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex_y <= max_y_coordinate
    ...     for vertex_x, vertex_y in contour)
    True
    """
    return (strategies.tuples(*repeat(points(x_coordinates, y_coordinates),
                                      times=3))
            .filter(is_contour_strict)
            .map(list))


def _validate_sizes(min_size: int, max_size: Optional[int],
                    min_expected_size: int) -> None:
    if max_size is None:
        pass
    elif max_size < min_expected_size:
        raise ValueError('Should have at least {expected} vertices, '
                         'but requested {actual}.'
                         .format(expected=min_expected_size,
                                 actual=max_size))
    elif min_size > max_size:
        raise ValueError('`min_size` should not be greater than `max_size`, '
                         'but found {min_size}, {max_size}.'
                         .format(min_size=min_size,
                                 max_size=max_size))
    if min_size < min_expected_size:
        warnings.warn('`min_size` is expected to be not less than {expected}, '
                      'but found {actual}.'
                      .format(expected=min_expected_size,
                              actual=min_size),
                      HypothesisWarning)


def _has_valid_size(sized: Sized,
                    *,
                    min_size: int,
                    max_size: Optional[int]) -> bool:
    size = len(sized)
    return min_size <= size and (max_size is None or size <= max_size)
