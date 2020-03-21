import warnings
from functools import partial
from itertools import (cycle,
                       groupby)
from random import Random
from typing import (List,
                    Optional,
                    Sized,
                    Tuple)

from hypothesis import strategies
from hypothesis.errors import HypothesisWarning

from .core.contracts import (is_contour_non_convex,
                             is_contour_strict,
                             points_do_not_lie_on_the_same_line)
from .core.utils import (Orientation,
                         orientation)
from .hints import (BoundingBox,
                    Contour,
                    Coordinate,
                    Point,
                    Polygon,
                    Polyline,
                    Segment,
                    Strategy)
from .utils import (constrict_convex_hull_size,
                    pack,
                    sort_pair,
                    to_concave_contour,
                    to_convex_contour,
                    to_convex_hull,
                    to_polygon,
                    to_strict_convex_hull)


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


TRIANGULAR_CONTOUR_SIZE = 3
MIN_CONCAVE_CONTOUR_SIZE = 4


def contours(x_coordinates: Strategy[Coordinate],
             y_coordinates: Optional[Strategy[Coordinate]] = None,
             *,
             min_size: int = TRIANGULAR_CONTOUR_SIZE,
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
    _validate_sizes(min_size, max_size, TRIANGULAR_CONTOUR_SIZE)
    if max_size is not None and max_size == TRIANGULAR_CONTOUR_SIZE:
        return triangular_contours(x_coordinates, y_coordinates)
    min_size = max(min_size, TRIANGULAR_CONTOUR_SIZE)
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
                    min_size: int = TRIANGULAR_CONTOUR_SIZE,
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
    _validate_sizes(min_size, max_size, TRIANGULAR_CONTOUR_SIZE)
    if max_size is not None and max_size == TRIANGULAR_CONTOUR_SIZE:
        return triangular_contours(x_coordinates, y_coordinates)
    min_size = max(min_size, TRIANGULAR_CONTOUR_SIZE)

    def to_points_with_random(points: List[Point]
                              ) -> Strategy[Tuple[List[Point], Random]]:
        return strategies.tuples(strategies.just(points), strategies.randoms())

    result = (strategies.lists(points(x_coordinates, y_coordinates),
                               min_size=min_size,
                               max_size=max_size,
                               unique=True)
              .flatmap(to_points_with_random)
              .map(pack(to_convex_contour))
              .filter(partial(_has_valid_size,
                              min_size=min_size,
                              max_size=max_size)))
    result = (rectangular_contours(x_coordinates, y_coordinates) | result
              if min_size <= RECTANGULAR_CONTOUR_SIZE
              else result)
    return (triangular_contours(x_coordinates, y_coordinates) | result
            if min_size == TRIANGULAR_CONTOUR_SIZE
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
    min_size = max(min_size, MIN_CONCAVE_CONTOUR_SIZE)

    def to_points_with_sizes(points: List[Point]
                             ) -> Strategy[Tuple[List[Point], int]]:
        sizes = strategies.integers(min_size,
                                    len(points)
                                    if max_size is None
                                    else min(len(points), max_size))
        return strategies.tuples(strategies.just(points), sizes)

    return (strategies.lists(points(x_coordinates, y_coordinates),
                             min_size=min_size,
                             max_size=max_size,
                             unique=True)
            .filter(points_do_not_lie_on_the_same_line)
            .flatmap(to_points_with_sizes)
            .map(pack(to_concave_contour))
            .filter(partial(_has_valid_size,
                            min_size=min_size,
                            max_size=max_size))
            .filter(is_contour_non_convex))


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
    vertices = points(x_coordinates, y_coordinates)

    def to_counterclockwise_contour(contour: Contour) -> Contour:
        return (contour
                if orientation(*contour) is Orientation.CLOCKWISE
                else contour[::-1])

    return (strategies.tuples(vertices, vertices, vertices)
            .filter(is_contour_strict)
            .map(list)
            .map(to_counterclockwise_contour))


RECTANGULAR_CONTOUR_SIZE = 4


def rectangular_contours(x_coordinates: Strategy[Coordinate],
                         y_coordinates: Optional[Strategy[Coordinate]] = None,
                         ) -> Strategy[Contour]:
    """
    Returns a strategy for axis-aligned rectangular contours.
    Rectangular contour is a contour formed by 4 points.

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
    >>> contours = planar.rectangular_contours(coordinates)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> len(contour) == 4
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
    >>> contours = planar.rectangular_contours(x_coordinates, y_coordinates)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> len(contour) == 4
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

    def bounding_box_to_contour(bounding_box: BoundingBox) -> Contour:
        x_min, x_max, y_min, y_max = bounding_box
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    return (bounding_boxes(x_coordinates, y_coordinates)
            .map(bounding_box_to_contour))


def bounding_boxes(x_coordinates: Strategy[Coordinate],
                   y_coordinates: Optional[Strategy[Coordinate]] = None,
                   ) -> Strategy[BoundingBox]:
    """
    Returns a strategy for bounding boxes.
    Bounding box is a quadruple of ``x_min``, ``x_max``, ``y_min``, ``y_max``.

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
    >>> bounding_boxes = planar.bounding_boxes(coordinates)
    >>> bounding_box = bounding_boxes.example()
    >>> isinstance(bounding_box, tuple)
    True
    >>> len(bounding_box) == 4
    True
    >>> all(isinstance(coordinate, coordinates_type)
    ...     for coordinate in bounding_box)
    True
    >>> all(min_coordinate <= coordinate <= max_coordinate
    ...     for coordinate in bounding_box)
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
    >>> bounding_boxes = planar.bounding_boxes(x_coordinates, y_coordinates)
    >>> bounding_box = bounding_boxes.example()
    >>> isinstance(bounding_box, tuple)
    True
    >>> len(bounding_box) == 4
    True
    >>> all(isinstance(coordinate, coordinates_type)
    ...     for coordinate in bounding_box)
    True
    >>> all(min_x_coordinate <= coordinate <= max_x_coordinate
    ...     for coordinate in bounding_box[:2])
    True
    >>> all(min_y_coordinate <= coordinate <= max_y_coordinate
    ...     for coordinate in bounding_box[2:])
    True
    """
    if y_coordinates is None:
        y_coordinates = x_coordinates

    def to_bounding_box(x_bounds: Tuple[Coordinate, Coordinate],
                        y_bounds: Tuple[Coordinate, Coordinate]
                        ) -> BoundingBox:
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        return x_min, x_max, y_min, y_max

    return strategies.builds(to_bounding_box,
                             strategies.lists(x_coordinates,
                                              min_size=2,
                                              max_size=2,
                                              unique=True)
                             .map(sort_pair),
                             strategies.lists(y_coordinates,
                                              min_size=2,
                                              max_size=2,
                                              unique=True)
                             .map(sort_pair))


EMPTY_HOLES_SIZE = 0


def polygons(x_coordinates: Strategy[Coordinate],
             y_coordinates: Optional[Strategy[Coordinate]] = None,
             *,
             min_size: int = TRIANGULAR_CONTOUR_SIZE,
             max_size: Optional[int] = None,
             min_holes_size: int = EMPTY_HOLES_SIZE,
             max_holes_size: Optional[int] = None,
             min_hole_size: int = TRIANGULAR_CONTOUR_SIZE,
             max_hole_size: Optional[int] = None) -> Strategy[Polygon]:
    """
    Returns a strategy for polygons.
    Polygon is a pair of contour (called polygon’s border)
    and possibly empty sequence of non-overlapping contours
    which lie within the border (called polygon’s holes).

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for border size.
    :param max_size: upper bound for border size, ``None`` for unbound.
    :param min_holes_size: lower bound for holes count.
    :param max_holes_size: upper bound for holes count, ``None`` for countless.
    :param min_hole_size: lower bound for hole size.
    :param max_hole_size: upper bound for hole size, ``None`` for unbound.

    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> min_holes_size, max_holes_size = 1, 4
    >>> min_hole_size, max_hole_size = 3, 5
    >>> polygons = planar.polygons(coordinates,
    ...                            min_size=min_size,
    ...                            max_size=max_size,
    ...                            min_holes_size=min_holes_size,
    ...                            max_holes_size=max_holes_size,
    ...                            min_hole_size=min_hole_size,
    ...                            max_hole_size=max_hole_size)
    >>> polygon = polygons.example()
    >>> isinstance(polygon, tuple)
    True
    >>> len(polygon) == 2
    True
    >>> border, holes = polygon
    >>> isinstance(border, list)
    True
    >>> isinstance(holes, list)
    True
    >>> all(isinstance(hole, list) for hole in holes)
    True
    >>> min_size <= len(border) <= max_size
    True
    >>> min_holes_size <= len(holes) <= max_holes_size
    True
    >>> all(min_hole_size <= len(hole) <= max_hole_size for hole in holes)
    True
    >>> all(isinstance(vertex, tuple) for vertex in border)
    True
    >>> all(isinstance(vertex, tuple) for hole in holes for vertex in hole)
    True
    >>> all(len(vertex) == 2 for vertex in border)
    True
    >>> all(len(vertex) == 2 for hole in holes for vertex in hole)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in border)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for hole in holes
    ...     for vertex in hole)
    True
    >>> all(all(min_coordinate <= coordinate <= max_coordinate
    ...         for coordinate in vertex)
    ...     for vertex in border)
    True
    >>> all(all(min_coordinate <= coordinate <= max_coordinate
    ...         for coordinate in vertex)
    ...     for hole in holes
    ...     for vertex in hole)
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
    >>> min_holes_size, max_holes_size = 1, 4
    >>> min_hole_size, max_hole_size = 3, 5
    >>> polygons = planar.polygons(x_coordinates, y_coordinates,
    ...                            min_size=min_size,
    ...                            max_size=max_size,
    ...                            min_holes_size=min_holes_size,
    ...                            max_holes_size=max_holes_size,
    ...                            min_hole_size=min_hole_size,
    ...                            max_hole_size=max_hole_size)
    >>> polygon = polygons.example()
    >>> isinstance(polygon, tuple)
    True
    >>> len(polygon) == 2
    True
    >>> border, holes = polygon
    >>> isinstance(border, list)
    True
    >>> isinstance(holes, list)
    True
    >>> all(isinstance(hole, list) for hole in holes)
    True
    >>> min_size <= len(border) <= max_size
    True
    >>> min_holes_size <= len(holes) <= max_holes_size
    True
    >>> all(min_hole_size <= len(hole) <= max_hole_size for hole in holes)
    True
    >>> all(isinstance(vertex, tuple) for vertex in border)
    True
    >>> all(isinstance(vertex, tuple) for hole in holes for vertex in hole)
    True
    >>> all(len(vertex) == 2 for vertex in border)
    True
    >>> all(len(vertex) == 2 for hole in holes for vertex in hole)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for vertex in border)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for coordinate in vertex)
    ...     for hole in holes
    ...     for vertex in hole)
    True
    >>> all(min_x_coordinate <= vertex_x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex_y <= max_y_coordinate
    ...     for vertex_x, vertex_y in border)
    True
    >>> all(min_x_coordinate <= vertex_x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex_y <= max_y_coordinate
    ...     for hole in holes
    ...     for vertex_x, vertex_y in hole)
    True
    """
    _validate_sizes(min_size, max_size, TRIANGULAR_CONTOUR_SIZE)
    _validate_sizes(min_holes_size, max_holes_size, EMPTY_HOLES_SIZE,
                    'min_holes_size', 'max_holes_size')
    _validate_sizes(min_hole_size, max_hole_size, TRIANGULAR_CONTOUR_SIZE,
                    'min_hole_size', 'max_hole_size')
    min_size, min_hole_size = (max(min_size, TRIANGULAR_CONTOUR_SIZE),
                               max(min_hole_size, TRIANGULAR_CONTOUR_SIZE))

    def to_points_with_sizes(points: List[Point]
                             ) -> Strategy[Tuple[List[Point], int, List[int]]]:
        max_border_points_count = len(points) - min_inner_points_count
        min_border_size = max(min_size, len(to_strict_convex_hull(points)))
        max_border_size = (max_border_points_count
                           if max_size is None
                           else min(max_size, max_border_points_count))
        return strategies.tuples(strategies.just(points),
                                 strategies.integers(min_border_size,
                                                     max_border_size),
                                 to_holes_sizes(points))

    def to_holes_sizes(points: List[Point]) -> Strategy[List[int]]:
        max_inner_points_count = len(points) - len(to_convex_hull(points))
        holes_size_scale = max_inner_points_count // min_hole_size
        points_max_hole_size = (holes_size_scale
                                if max_holes_size is None
                                else min(max_holes_size, holes_size_scale))
        return (strategies.integers(min_holes_size, points_max_hole_size)
                .flatmap(partial(_to_holes_sizes,
                                 min_hole_size=min_hole_size,
                                 max_hole_size=max_inner_points_count))
                if max_inner_points_count >= min_hole_size
                else strategies.builds(list))

    def _to_holes_sizes(holes_size: int,
                        min_hole_size: int,
                        max_hole_size: int) -> Strategy[List[int]]:
        if not holes_size:
            return strategies.builds(list)
        max_hole_size -= holes_size * min_hole_size
        max_holes_sizes = [min_hole_size] * holes_size
        indices = cycle(range(holes_size))
        while max_hole_size:
            max_holes_sizes[next(indices)] += 1
            max_hole_size -= 1
        sizes_ranges = [range(min_hole_size, max_hole_size + 1)
                        for max_hole_size in max_holes_sizes]
        return (strategies.permutations([strategies.sampled_from(sizes_range)
                                         for sizes_range in sizes_ranges])
                .flatmap(pack(strategies.tuples))
                .map(list))

    def has_valid_sizes(polygon: Polygon) -> bool:
        border, holes = polygon
        return (_has_valid_size(border,
                                min_size=min_size,
                                max_size=max_size)
                and _has_valid_size(holes,
                                    min_size=min_holes_size,
                                    max_size=max_holes_size)
                and all(_has_valid_size(hole,
                                        min_size=min_hole_size,
                                        max_size=max_hole_size)
                        for hole in holes))

    min_inner_points_count = min_hole_size * min_holes_size

    def has_valid_inner_points_count(points: List[Point]) -> bool:
        return ((max_size is None
                 or len(to_strict_convex_hull(points)) <= max_size)
                and (len(points) - len(to_convex_hull(points))
                     >= min_inner_points_count))

    return (strategies.lists(
            points(x_coordinates, y_coordinates),
            min_size=min_size + min_inner_points_count,
            max_size=(None
                      if (max_size is None
                          or max_holes_size is None
                          or max_hole_size is None)
                      else max_size + max_hole_size * max_holes_size),
            unique=True)
            .map(sorted)
            .map(partial(constrict_convex_hull_size,
                         max_size=max_size))
            .filter(has_valid_inner_points_count)
            .flatmap(to_points_with_sizes)
            .map(pack(to_polygon))
            .filter(has_valid_sizes))


def _validate_sizes(min_size: int, max_size: Optional[int],
                    min_expected_size: int,
                    min_size_name: str = 'min_size',
                    max_size_name: str = 'max_size') -> None:
    if max_size is None:
        pass
    elif max_size < min_expected_size:
        raise ValueError('`{max_size_name}` '
                         'should not be less than {min_expected_size}, '
                         'but found {max_size}.'
                         .format(max_size_name=max_size_name,
                                 min_expected_size=min_expected_size,
                                 max_size=max_size))
    elif min_size > max_size:
        raise ValueError('`{min_size_name}` '
                         'should not be greater than `{max_size_name}`, '
                         'but found {min_size}, {max_size}.'
                         .format(min_size_name=min_size_name,
                                 max_size_name=max_size_name,
                                 min_size=min_size,
                                 max_size=max_size))
    elif min_size < 0:
        raise ValueError('`{min_size_name}` '
                         'should not be less than 0, '
                         'but found {min_size}.'
                         .format(min_size_name=min_size_name,
                                 min_size=min_size))
    if min_size < min_expected_size:
        warnings.warn('`{min_size_name}` is expected to be '
                      'not less than {min_expected_size}, '
                      'but found {min_size}.'
                      .format(min_size_name=min_size_name,
                              min_expected_size=min_expected_size,
                              min_size=min_size),
                      HypothesisWarning)


def _has_valid_size(sized: Sized,
                    *,
                    min_size: int,
                    max_size: Optional[int]) -> bool:
    size = len(sized)
    return min_size <= size and (max_size is None or size <= max_size)
