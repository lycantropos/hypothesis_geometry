import warnings
from functools import partial
from itertools import (chain,
                       cycle,
                       groupby,
                       repeat)
from operator import add
from typing import (Callable,
                    List,
                    Optional,
                    Sequence,
                    Sized,
                    Tuple,
                    Type)

from ground.base import (Orientation,
                         get_context)
from ground.hints import (Box,
                          Contour,
                          Coordinate,
                          Multipoint,
                          Multisegment,
                          Point,
                          Segment)
from hypothesis import strategies
from hypothesis.errors import HypothesisWarning

from .core.contracts import (are_segments_non_crossing_non_overlapping,
                             has_horizontal_lowermost_segment,
                             has_vertical_leftmost_segment,
                             to_non_collinear_points_detector,
                             to_non_convex_vertices_detector,
                             to_strict_vertices_detector)
from .core.factories import (to_contour_edges_constructor,
                             to_convex_hull_size_constrictor,
                             to_convex_vertices_sequence_factory,
                             to_max_convex_hull_constructor,
                             to_multicontour_factory,
                             to_polygon_border_edges_constructor,
                             to_polygon_factory,
                             to_star_contour_vertices_factory,
                             to_vertices_sequence_factory)
from .core.hints import (Chooser,
                         Domain,
                         Orienteer,
                         PointsSequenceOperator,
                         PolygonEdgesConstructor)
from .core.utils import (pack,
                         pairwise,
                         sort_pair)
from .hints import (Mix,
                    Multicontour,
                    Multipolygon,
                    Polygon,
                    Polyline,
                    Strategy)


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

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Point = context.point_cls

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> points = planar.points(coordinates)
    >>> point = points.example()
    >>> isinstance(point, Point)
    True
    >>> (isinstance(point.x, coordinates_type)
    ...  and isinstance(point.y, coordinates_type))
    True
    >>> (min_coordinate <= point.x <= max_coordinate
    ...  and min_coordinate <= point.y <= max_coordinate)
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
    >>> isinstance(point, Point)
    True
    >>> (isinstance(point.x, coordinates_type)
    ...  and isinstance(point.y, coordinates_type))
    True
    >>> (min_x_coordinate <= point.x <= max_x_coordinate
    ...  and min_y_coordinate <= point.y <= max_y_coordinate)
    True
    """
    return strategies.builds(get_context().point_cls,
                             x_coordinates,
                             x_coordinates
                             if y_coordinates is None
                             else y_coordinates)


EMPTY_MULTIPOINT_SIZE = 0


def multipoints(x_coordinates: Strategy[Coordinate],
                y_coordinates: Optional[Strategy[Coordinate]] = None,
                *,
                min_size: int = EMPTY_MULTIPOINT_SIZE,
                max_size: Optional[int] = None) -> Strategy[Multipoint]:
    """
    Returns a strategy for multipoints.
    Multipoint is a possibly empty sequence of distinct points.

    :param x_coordinates: strategy for points' x-coordinates.
    :param y_coordinates:
        strategy for points' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for multipoint size.
    :param max_size: upper bound for multipoint size, ``None`` for unbound.

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Multipoint = context.multipoint_cls

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> multipoints = planar.multipoints(coordinates,
    ...                                  min_size=min_size,
    ...                                  max_size=max_size)
    >>> multipoint = multipoints.example()
    >>> isinstance(multipoint, Multipoint)
    True
    >>> min_size <= len(multipoint.points) <= max_size
    True
    >>> all(isinstance(point.x, coordinates_type)
    ...     and isinstance(point.y, coordinates_type)
    ...     for point in multipoint.points)
    True
    >>> all(min_coordinate <= point.x <= max_coordinate
    ...     and min_coordinate <= point.y <= max_coordinate
    ...     for point in multipoint.points)
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
    >>> multipoints = planar.multipoints(x_coordinates, y_coordinates,
    ...                                      min_size=min_size,
    ...                                      max_size=max_size)
    >>> multipoint = multipoints.example()
    >>> isinstance(multipoint, Multipoint)
    True
    >>> min_size <= len(multipoint.points) <= max_size
    True
    >>> all(isinstance(point.x, coordinates_type)
    ...     and isinstance(point.y, coordinates_type)
    ...     for point in multipoint.points)
    True
    >>> all(min_x_coordinate <= point.x <= max_x_coordinate
    ...     and min_y_coordinate <= point.y <= max_y_coordinate
    ...     for point in multipoint.points)
    True
    """
    _validate_sizes(min_size, max_size, EMPTY_MULTIPOINT_SIZE)
    return (_unique_points_sequences(x_coordinates, y_coordinates,
                                     min_size=max(min_size,
                                                  EMPTY_MULTIPOINT_SIZE),
                                     max_size=max_size)
            .map(get_context().multipoint_cls))


def segments(x_coordinates: Strategy[Coordinate],
             y_coordinates: Optional[Strategy[Coordinate]] = None
             ) -> Strategy[Segment]:
    """
    Returns a strategy for segments.

    :param x_coordinates: strategy for endpoints' x-coordinates.
    :param y_coordinates:
        strategy for endpoints' y-coordinates,
        ``None`` for reusing x-coordinates strategy.

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Segment = context.segment_cls

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> segments = planar.segments(coordinates)
    >>> segment = segments.example()
    >>> isinstance(segment, Segment)
    True
    >>> (isinstance(segment.start.x, coordinates_type)
    ...  and isinstance(segment.start.y, coordinates_type)
    ...  and isinstance(segment.end.x, coordinates_type)
    ...  and isinstance(segment.end.y, coordinates_type))
    True
    >>> (min_coordinate <= segment.start.x <= max_coordinate
    ...  and min_coordinate <= segment.start.y <= max_coordinate
    ...  and min_coordinate <= segment.end.x <= max_coordinate
    ...  and min_coordinate <= segment.end.y <= max_coordinate)
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
    >>> isinstance(segment, Segment)
    True
    >>> (isinstance(segment.start.x, coordinates_type)
    ...  and isinstance(segment.start.y, coordinates_type)
    ...  and isinstance(segment.end.x, coordinates_type)
    ...  and isinstance(segment.end.y, coordinates_type))
    True
    >>> (min_x_coordinate <= segment.start.x <= max_x_coordinate
    ...  and min_y_coordinate <= segment.start.y <= max_y_coordinate
    ...  and min_x_coordinate <= segment.end.x <= max_x_coordinate
    ...  and min_y_coordinate <= segment.end.y <= max_y_coordinate)
    True
    """

    def non_degenerate_endpoints(endpoints: Tuple[Point, Point]) -> bool:
        start, end = endpoints
        return start != end

    points_strategy = points(x_coordinates, y_coordinates)
    return (strategies.tuples(points_strategy, points_strategy)
            .filter(non_degenerate_endpoints)
            .map(pack(get_context().segment_cls)))


EMPTY_MULTISEGMENT_SIZE = 0


def multisegments(x_coordinates: Strategy[Coordinate],
                  y_coordinates: Optional[Strategy[Coordinate]] = None,
                  *,
                  min_size: int = EMPTY_MULTISEGMENT_SIZE,
                  max_size: Optional[int] = None) -> Strategy[Multisegment]:
    """
    Returns a strategy for multisegments.
    Multisegment is a possibly empty sequence of segments
    such that any pair of them do not cross/overlap each other.

    :param x_coordinates: strategy for segments' x-coordinates.
    :param y_coordinates:
        strategy for segments' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for multisegment size.
    :param max_size: upper bound for multisegment size, ``None`` for unbound.

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Multisegment = context.multisegment_cls

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> multisegments = planar.multisegments(coordinates,
    ...                                      min_size=min_size,
    ...                                      max_size=max_size)
    >>> multisegment = multisegments.example()
    >>> isinstance(multisegment, Multisegment)
    True
    >>> min_size <= len(multisegment.segments) <= max_size
    True
    >>> all(isinstance(segment.start.x, coordinates_type)
    ...     and isinstance(segment.start.y, coordinates_type)
    ...     and isinstance(segment.start.x, coordinates_type)
    ...     and isinstance(segment.start.y, coordinates_type)
    ...     for segment in multisegment.segments)
    True
    >>> all(min_coordinate <= segment.start.x <= max_coordinate
    ...     and min_coordinate <= segment.start.y <= max_coordinate
    ...     and min_coordinate <= segment.end.x <= max_coordinate
    ...     and min_coordinate <= segment.end.y <= max_coordinate
    ...     for segment in multisegment.segments)
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
    >>> multisegments = planar.multisegments(x_coordinates, y_coordinates,
    ...                                      min_size=min_size,
    ...                                      max_size=max_size)
    >>> multisegment = multisegments.example()
    >>> isinstance(multisegment, Multisegment)
    True
    >>> min_size <= len(multisegment.segments) <= max_size
    True
    >>> all(isinstance(segment.start.x, coordinates_type)
    ...     and isinstance(segment.start.y, coordinates_type)
    ...     and isinstance(segment.start.x, coordinates_type)
    ...     and isinstance(segment.start.y, coordinates_type)
    ...     for segment in multisegment.segments)
    True
    >>> all(min_x_coordinate <= segment.start.x <= max_x_coordinate
    ...     and min_y_coordinate <= segment.start.y <= max_y_coordinate
    ...     and min_x_coordinate <= segment.end.x <= max_x_coordinate
    ...     and min_y_coordinate <= segment.end.y <= max_y_coordinate
    ...     for segment in multisegment.segments)
    True
    """
    _validate_sizes(min_size, max_size, EMPTY_MULTISEGMENT_SIZE)
    min_size = max(min_size, EMPTY_MULTISEGMENT_SIZE)
    return (_to_non_crossing_non_overlapping_segments_sequences(
            x_coordinates,
            y_coordinates,
            min_size=min_size,
            max_size=max_size)
            .map(get_context().multisegment_cls))


def _to_non_crossing_non_overlapping_segments_sequences(
        x_coordinates: Strategy[Coordinate],
        y_coordinates: Optional[Strategy[Coordinate]],
        *,
        min_size: int,
        max_size: Optional[int]) -> Strategy[Sequence[Segment]]:
    if y_coordinates is None:
        y_coordinates = x_coordinates
    if max_size is not None and max_size < 2:
        return (segments(x_coordinates, y_coordinates)
                .map(lambda segment: [segment])
                if max_size
                else strategies.builds(list))
    context = get_context()
    point_cls, segment_cls = context.point_cls, context.segment_cls

    def to_vertical_multisegment(x: Coordinate,
                                 ys: List[Coordinate]) -> Sequence[Segment]:
        return [segment_cls(point_cls(x, y), point_cls(x, next_y))
                for y, next_y in pairwise(sorted(ys))]

    def to_horizontal_multisegment(xs: List[Coordinate],
                                   y: Coordinate) -> Sequence[Segment]:
        return [segment_cls(point_cls(x, y), point_cls(next_x, y))
                for x, next_x in pairwise(sorted(xs))]

    next_min_size, next_max_size = (min_size + 1, (max_size
                                                   if max_size is None
                                                   else max_size + 1))
    result = (strategies.builds(to_vertical_multisegment,
                                x_coordinates,
                                strategies.lists(y_coordinates,
                                                 min_size=next_min_size,
                                                 max_size=next_max_size,
                                                 unique=True))
              | strategies.builds(to_horizontal_multisegment,
                                  strategies.lists(x_coordinates,
                                                   min_size=next_min_size,
                                                   max_size=next_max_size,
                                                   unique=True),
                                  y_coordinates))
    if min_size >= TRIANGULAR_CONTOUR_SIZE:
        result |= (_vertices_sequences(x_coordinates, y_coordinates,
                                       min_size=min_size,
                                       max_size=max_size)
                   .map(to_contour_edges_constructor(context))
                   .flatmap(partial(_to_sub_lists,
                                    min_size=min_size)))
    return result | (strategies.lists(segments(x_coordinates, y_coordinates),
                                      min_size=min_size,
                                      max_size=max_size)
                     .filter(are_segments_non_crossing_non_overlapping))


def _to_sub_lists(values: Sequence[Domain],
                  *,
                  min_size: int) -> Strategy[List[Domain]]:
    return strategies.builds(_cut,
                             strategies.permutations(values),
                             strategies.integers(min_size, len(values)))


def _cut(values: Domain, limit: int) -> Domain:
    return values[:limit] if limit < len(values) else values


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

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Point = context.point_cls

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
    >>> all(isinstance(vertex, Point) for vertex in polyline)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in polyline)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
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
    >>> all(isinstance(vertex, Point) for vertex in polyline)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in polyline)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for vertex in polyline)
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
                              max_size
                              if max_size is None
                              else max_size - 1)
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

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Point = context.point_cls

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> contours = planar.contours(coordinates,
    ...                            min_size=min_size,
    ...                            max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, Point) for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
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
    >>> all(isinstance(vertex, Point) for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for vertex in contour)
    True
    """
    return (_vertices_sequences(x_coordinates, y_coordinates,
                                min_size=min_size,
                                max_size=max_size)
            .map(get_context().contour_cls))


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

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Point = context.point_cls

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
    >>> all(isinstance(vertex, Point) for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
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
    >>> contours = planar.convex_contours(x_coordinates, y_coordinates,
    ...                                   min_size=min_size,
    ...                                   max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, Point) for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for vertex in contour)
    True
    """
    return (_convex_vertices_sequences(x_coordinates, y_coordinates,
                                       min_size=min_size,
                                       max_size=max_size)
            .map(get_context().contour_cls))


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

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Contour = context.contour_cls

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
    >>> isinstance(contour, Contour)
    True
    >>> min_size <= len(contour.vertices) <= max_size
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour.vertices)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
    ...     for vertex in contour.vertices)
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
    >>> contours = planar.concave_contours(x_coordinates, y_coordinates,
    ...                                    min_size=min_size,
    ...                                    max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, Contour)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, Point) for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour.vertices)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for vertex in contour.vertices)
    True
    """
    return (_concave_vertices_sequences(x_coordinates, y_coordinates,
                                        min_size=min_size,
                                        max_size=max_size)
            .map(get_context().contour_cls))


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

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Contour = context.contour_cls

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> contours = planar.triangular_contours(coordinates)
    >>> contour = contours.example()
    >>> isinstance(contour, Contour)
    True
    >>> len(contour.vertices) == 3
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour.vertices)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
    ...     for vertex in contour.vertices)
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
    >>> isinstance(contour, Contour)
    True
    >>> len(contour) == 3
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour.vertices)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for vertex in contour.vertices)
    True
    """
    return (_triangular_vertices_sequences(x_coordinates, y_coordinates)
            .map(get_context().contour_cls))


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

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Point = context.point_cls

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
    >>> all(isinstance(vertex, Point) for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
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
    >>> all(isinstance(vertex, Point) for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for vertex in contour)
    True
    """
    return (_rectangular_vertices_sequences(x_coordinates, y_coordinates)
            .map(get_context().contour_cls))


def boxes(x_coordinates: Strategy[Coordinate],
          y_coordinates: Optional[Strategy[Coordinate]] = None
          ) -> Strategy[Box]:
    """
    Returns a strategy for boxes.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Box = context.box_cls

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> boxes = planar.boxes(coordinates)
    >>> box = boxes.example()
    >>> isinstance(box, Box)
    True
    >>> (isinstance(box.min_x, coordinates_type)
    ...  and isinstance(box.max_x, coordinates_type)
    ...  and isinstance(box.min_y, coordinates_type)
    ...  and isinstance(box.max_y, coordinates_type))
    True
    >>> (min_coordinate <= box.min_x <= max_coordinate
    ...  and min_coordinate <= box.max_x <= max_coordinate
    ...  and min_coordinate <= box.min_y <= max_coordinate
    ...  and min_coordinate <= box.max_y <= max_coordinate)
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
    >>> boxes = planar.boxes(x_coordinates, y_coordinates)
    >>> box = boxes.example()
    >>> isinstance(box, Box)
    True
    >>> (isinstance(box.min_x, coordinates_type)
    ...  and isinstance(box.max_x, coordinates_type)
    ...  and isinstance(box.min_y, coordinates_type)
    ...  and isinstance(box.max_y, coordinates_type))
    True
    >>> (min_x_coordinate <= box.min_x <= max_x_coordinate
    ...  and min_x_coordinate <= box.max_x <= max_x_coordinate)
    True
    >>> (min_y_coordinate <= box.min_y <= max_y_coordinate
    ...  and min_y_coordinate <= box.max_y <= max_y_coordinate)
    True
    """
    return (strategies.builds(add,
                              strategies.lists(x_coordinates,
                                               min_size=2,
                                               max_size=2,
                                               unique=True)
                              .map(sort_pair),
                              strategies.lists(x_coordinates
                                               if y_coordinates is None
                                               else y_coordinates,
                                               min_size=2,
                                               max_size=2,
                                               unique=True)
                              .map(sort_pair))
            .map(pack(get_context().box_cls)))


def star_contours(x_coordinates: Strategy[Coordinate],
                  y_coordinates: Optional[Strategy[Coordinate]] = None,
                  *,
                  min_size: int = TRIANGULAR_CONTOUR_SIZE,
                  max_size: Optional[int] = None) -> Strategy[Contour]:
    """
    Returns a strategy for star contours.
    Star contour is a contour such that every vertex is visible from centroid,
    i.e. segments from centroid to vertices do not cross or overlap contour.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for contour size.
    :param max_size: upper bound for contour size, ``None`` for unbound.

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Point = context.point_cls

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> contours = planar.star_contours(coordinates,
    ...                                 min_size=min_size,
    ...                                 max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, Point) for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
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
    >>> contours = planar.star_contours(x_coordinates, y_coordinates,
    ...                                 min_size=min_size,
    ...                                 max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, list)
    True
    >>> min_size <= len(contour) <= max_size
    True
    >>> all(isinstance(vertex, Point) for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in contour)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for vertex in contour)
    True
    """
    return (_star_vertices_sequences(x_coordinates, y_coordinates,
                                     min_size=min_size,
                                     max_size=max_size)
            .map(get_context().contour_cls))


EMPTY_MULTICONTOUR_SIZE = 0


def multicontours(x_coordinates: Strategy[Coordinate],
                  y_coordinates: Optional[Strategy[Coordinate]] = None,
                  *,
                  min_size: int = EMPTY_MULTICONTOUR_SIZE,
                  max_size: Optional[int] = None,
                  min_contour_size: int = TRIANGULAR_CONTOUR_SIZE,
                  max_contour_size: Optional[int] = None
                  ) -> Strategy[Multicontour]:
    """
    Returns a strategy for multicontours.
    Multicontour is a possibly empty sequence of non-crossing
    and non-overlapping contours.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for size.
    :param max_size: upper bound for size, ``None`` for unbound.
    :param min_contour_size: lower bound for contour size.
    :param max_contour_size:
        upper bound for contour size, ``None`` for unbound.

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Point = context.point_cls

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 5, 10
    >>> min_contour_size, max_contour_size = 3, 5
    >>> multicontours = planar.multicontours(coordinates,
    ...                                      min_size=min_size,
    ...                                      max_size=max_size,
    ...                                      min_contour_size=min_contour_size,
    ...                                      max_contour_size=max_contour_size)
    >>> multicontour = multicontours.example()
    >>> isinstance(multicontour, list)
    True
    >>> all(isinstance(contour, list) for contour in multicontour)
    True
    >>> min_size <= len(multicontour) <= max_size
    True
    >>> all(min_contour_size <= len(contour) <= max_contour_size
    ...     for contour in multicontour)
    True
    >>> all(isinstance(vertex, Point)
    ...     for contour in multicontour
    ...     for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for contour in multicontour
    ...     for vertex in contour)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
    ...     for contour in multicontour
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
    >>> min_contour_size, max_contour_size = 3, 5
    >>> multicontours = planar.multicontours(x_coordinates, y_coordinates,
    ...                                      min_size=min_size,
    ...                                      max_size=max_size,
    ...                                      min_contour_size=min_contour_size,
    ...                                      max_contour_size=max_contour_size)
    >>> multicontour = multicontours.example()
    >>> isinstance(multicontour, list)
    True
    >>> all(isinstance(contour, list) for contour in multicontour)
    True
    >>> min_size <= len(multicontour) <= max_size
    True
    >>> all(min_contour_size <= len(contour) <= max_contour_size
    ...     for contour in multicontour)
    True
    >>> all(isinstance(vertex, Point)
    ...     for contour in multicontour
    ...     for vertex in contour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for contour in multicontour
    ...     for vertex in contour)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for contour in multicontour
    ...     for vertex in contour)
    True
    """
    _validate_sizes(min_size, max_size, EMPTY_MULTICONTOUR_SIZE)
    _validate_sizes(min_contour_size, max_contour_size,
                    TRIANGULAR_CONTOUR_SIZE,
                    'min_contour_size', 'max_contour_size')
    min_size, min_contour_size = (max(min_size, EMPTY_MULTICONTOUR_SIZE),
                                  max(min_contour_size,
                                      TRIANGULAR_CONTOUR_SIZE))

    def to_multicontours(vertices: List[Point]) -> Strategy[Multicontour]:
        return strategies.builds(to_multicontour_factory(get_context()),
                                 strategies.just(vertices),
                                 to_sizes(len(vertices)),
                                 _choosers())

    def to_sizes(limit: int) -> Strategy[List[int]]:
        return (strategies.integers(min_size, limit // min_contour_size)
                .flatmap(partial(_to_sizes,
                                 min_element_size=min_contour_size,
                                 limit=limit)))

    def _to_sizes(size: int,
                  min_element_size: int,
                  limit: int) -> Strategy[List[int]]:
        if not size:
            return strategies.builds(list)
        max_sizes = [min_element_size] * size
        indices = cycle(range(size))
        for _ in range(limit - size * min_element_size):
            max_sizes[next(indices)] += 1
        sizes_ranges = [range(min_element_size, max_element_size + 1)
                        for max_element_size in max_sizes]
        return (strategies.permutations([strategies.sampled_from(sizes_range)
                                         for sizes_range in sizes_ranges])
                .flatmap(pack(strategies.tuples))
                .map(list))

    def has_valid_sizes(multicontour: Multicontour) -> bool:
        return (_has_valid_size(multicontour,
                                min_size=min_size,
                                max_size=max_size)
                and all(_has_valid_size(contour.vertices,
                                        min_size=min_contour_size,
                                        max_size=max_contour_size)
                        for contour in multicontour))

    return (strategies.lists(points(x_coordinates, y_coordinates),
                             min_size=min_size * min_contour_size,
                             max_size=(None
                                       if (max_size is None
                                           or max_contour_size is None)
                                       else max_size * max_contour_size),
                             unique=True)
            .flatmap(to_multicontours)
            .filter(has_valid_sizes))


def polygons(x_coordinates: Strategy[Coordinate],
             y_coordinates: Optional[Strategy[Coordinate]] = None,
             *,
             min_size: int = TRIANGULAR_CONTOUR_SIZE,
             max_size: Optional[int] = None,
             min_holes_size: int = EMPTY_MULTICONTOUR_SIZE,
             max_holes_size: Optional[int] = None,
             min_hole_size: int = TRIANGULAR_CONTOUR_SIZE,
             max_hole_size: Optional[int] = None) -> Strategy[Polygon]:
    """
    Returns a strategy for polygons.
    Polygon is a pair of contour (called polygon’s border)
    and multicontour which lies within the border (called polygon’s holes).

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

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Point = context.point_cls

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
    >>> all(isinstance(vertex, Point) for vertex in border)
    True
    >>> all(isinstance(vertex, Point) for hole in holes for vertex in hole)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in border)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for hole in holes
    ...     for vertex in hole)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
    ...     for vertex in border)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
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
    >>> all(isinstance(vertex, Point) for vertex in border)
    True
    >>> all(isinstance(vertex, Point) for hole in holes for vertex in hole)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in border)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for hole in holes
    ...     for vertex in hole)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for vertex in border)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for hole in holes
    ...     for vertex in hole)
    True
    """
    _validate_sizes(min_size, max_size, TRIANGULAR_CONTOUR_SIZE)
    _validate_sizes(min_holes_size, max_holes_size, EMPTY_MULTICONTOUR_SIZE,
                    'min_holes_size', 'max_holes_size')
    _validate_sizes(min_hole_size, max_hole_size, TRIANGULAR_CONTOUR_SIZE,
                    'min_hole_size', 'max_hole_size')
    min_size, min_hole_size, min_holes_size = (
        max(min_size, TRIANGULAR_CONTOUR_SIZE),
        max(min_hole_size, TRIANGULAR_CONTOUR_SIZE),
        max(min_holes_size, EMPTY_MULTICONTOUR_SIZE))
    context = get_context()

    def to_polygons(points: List[Point]) -> Strategy[Polygon]:
        max_border_points_count = len(points) - min_inner_points_count
        min_border_size = max(min_size,
                              len(context.points_convex_hull(points)))
        max_border_size = (max_border_points_count
                           if max_size is None
                           else min(max_size, max_border_points_count))
        return strategies.builds(
                to_polygon_factory(context),
                strategies.just(points),
                strategies.integers(min_border_size,
                                    max_border_size),
                to_holes_sizes(to_max_convex_hull_constructor(context),
                               points),
                _choosers())

    def to_holes_sizes(max_convex_hull_constructor: PointsSequenceOperator,
                       points: List[Point]) -> Strategy[List[int]]:
        max_inner_points_count = (len(points)
                                  - len(max_convex_hull_constructor(points)))
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
        max_holes_sizes = [min_hole_size] * holes_size
        indices = cycle(range(holes_size))
        for _ in range(max_hole_size - holes_size * min_hole_size):
            max_holes_sizes[next(indices)] += 1
        sizes_ranges = [range(min_hole_size, max_hole_size + 1)
                        for max_hole_size in max_holes_sizes]
        return (strategies.permutations([strategies.sampled_from(sizes_range)
                                         for sizes_range in sizes_ranges])
                .flatmap(pack(strategies.tuples))
                .map(list))

    def has_valid_sizes(polygon: Polygon) -> bool:
        border, holes = polygon
        return (_has_valid_size(border.vertices,
                                min_size=min_size,
                                max_size=max_size)
                and _has_valid_size(holes,
                                    min_size=min_holes_size,
                                    max_size=max_holes_size)
                and all(_has_valid_size(hole.vertices,
                                        min_size=min_hole_size,
                                        max_size=max_hole_size)
                        for hole in holes))

    min_inner_points_count = min_hole_size * min_holes_size

    def has_valid_inner_points_count(convex_hull_constructor
                                     : PointsSequenceOperator,
                                     max_convex_hull_constructor
                                     : PointsSequenceOperator,
                                     points_sequence: Sequence[Point]) -> bool:
        return ((max_size is None
                 or len(convex_hull_constructor(points_sequence)) <= max_size)
                and (len(points_sequence)
                     - len(max_convex_hull_constructor(points_sequence))
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
            .filter(to_non_collinear_points_detector(context))
            .map(sorted)
            .map(to_convex_hull_size_constrictor(context,
                                                 max_size=max_size))
            .filter(partial(has_valid_inner_points_count,
                            context.points_convex_hull,
                            to_max_convex_hull_constructor(context)))
            .flatmap(to_polygons)
            .filter(has_valid_sizes))


EMPTY_MULTIPOLYGON_SIZE = 0


def multipolygons(x_coordinates: Strategy[Coordinate],
                  y_coordinates: Optional[Strategy[Coordinate]] = None,
                  *,
                  min_size: int = EMPTY_MULTIPOLYGON_SIZE,
                  max_size: Optional[int] = None,
                  min_border_size: int = TRIANGULAR_CONTOUR_SIZE,
                  max_border_size: Optional[int] = None,
                  min_holes_size: int = EMPTY_MULTICONTOUR_SIZE,
                  max_holes_size: Optional[int] = None,
                  min_hole_size: int = TRIANGULAR_CONTOUR_SIZE,
                  max_hole_size: Optional[int] = None
                  ) -> Strategy[Multipolygon]:
    """
    Returns a strategy for multipolygons.
    Multipolygon is a possibly empty sequence of polygons
    with non-crossing and non-overlapping borders.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_size: lower bound for size.
    :param max_size: upper bound for size, ``None`` for unbound.
    :param min_border_size: lower bound for polygons' border size.
    :param max_border_size:
        upper bound for polygons' border size, ``None`` for unbound.
    :param min_holes_size: lower bound for polygons' holes count.
    :param max_holes_size:
        upper bound for polygons' holes count, ``None`` for countless.
    :param min_hole_size: lower bound for hole size.
    :param max_hole_size:
        upper bound for polygons' hole size, ``None`` for unbound.

    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_size, max_size = 0, 5
    >>> min_border_size, max_border_size = 5, 10
    >>> min_holes_size, max_holes_size = 1, 4
    >>> min_hole_size, max_hole_size = 3, 5
    >>> multipolygons = planar.multipolygons(coordinates,
    ...                                      min_size=min_size,
    ...                                      max_size=max_size,
    ...                                      min_border_size=min_border_size,
    ...                                      max_border_size=max_border_size,
    ...                                      min_holes_size=min_holes_size,
    ...                                      max_holes_size=max_holes_size,
    ...                                      min_hole_size=min_hole_size,
    ...                                      max_hole_size=max_hole_size)
    >>> multipolygon = multipolygons.example()
    >>> isinstance(multipolygon, list)
    True
    >>> min_size <= len(multipolygon) <= max_size
    True
    >>> all(isinstance(polygon, tuple) for polygon in multipolygon)
    True
    >>> all(len(polygon) == 2 for polygon in multipolygon)
    True
    >>> all(isinstance(border, list)
    ...     and isinstance(holes, list)
    ...     and all(isinstance(hole, list) for hole in holes)
    ...     for border, holes in multipolygon)
    True
    >>> all(min_border_size <= len(border) <= max_border_size
    ...     and min_holes_size <= len(holes) <= max_holes_size
    ...     and all(min_hole_size <= len(hole) <= max_hole_size
    ...             for hole in holes)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(isinstance(vertex, tuple) for vertex in border)
    ...     and all(isinstance(vertex, tuple)
    ...             for hole in holes
    ...             for vertex in hole)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(len(vertex) == 2 for vertex in border)
    ...     and all(len(vertex) == 2 for hole in holes for vertex in hole)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for vertex in border
    ...         for coordinate in vertex)
    ...     and all(isinstance(coordinate, coordinates_type)
    ...             for hole in holes
    ...             for vertex in hole
    ...             for coordinate in vertex)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(all(min_coordinate <= coordinate <= max_coordinate
    ...             for coordinate in vertex)
    ...         for vertex in border)
    ...     and all(min_coordinate <= coordinate <= max_coordinate
    ...             for hole in holes
    ...             for vertex in hole
    ...             for coordinate in vertex)
    ...     for border, holes in multipolygon)
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
    >>> min_size, max_size = 0, 5
    >>> min_border_size, max_border_size = 5, 10
    >>> min_holes_size, max_holes_size = 1, 4
    >>> min_hole_size, max_hole_size = 3, 5
    >>> multipolygons = planar.multipolygons(x_coordinates, y_coordinates,
    ...                                      min_size=min_size,
    ...                                      max_size=max_size,
    ...                                      min_border_size=min_border_size,
    ...                                      max_border_size=max_border_size,
    ...                                      min_holes_size=min_holes_size,
    ...                                      max_holes_size=max_holes_size,
    ...                                      min_hole_size=min_hole_size,
    ...                                      max_hole_size=max_hole_size)
    >>> multipolygon = multipolygons.example()
    >>> isinstance(multipolygon, list)
    True
    >>> min_size <= len(multipolygon) <= max_size
    True
    >>> all(isinstance(polygon, tuple) for polygon in multipolygon)
    True
    >>> all(len(polygon) == 2 for polygon in multipolygon)
    True
    >>> all(isinstance(border, list)
    ...     and isinstance(holes, list)
    ...     and all(isinstance(hole, list) for hole in holes)
    ...     for border, holes in multipolygon)
    True
    >>> all(min_border_size <= len(border) <= max_border_size
    ...     and min_holes_size <= len(holes) <= max_holes_size
    ...     and all(min_hole_size <= len(hole) <= max_hole_size
    ...             for hole in holes)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(isinstance(vertex, tuple) for vertex in border)
    ...     and all(isinstance(vertex, tuple)
    ...             for hole in holes
    ...             for vertex in hole)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(len(vertex) == 2 for vertex in border)
    ...     and all(len(vertex) == 2 for hole in holes for vertex in hole)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(isinstance(coordinate, coordinates_type)
    ...         for vertex in border
    ...         for coordinate in vertex)
    ...     and all(isinstance(coordinate, coordinates_type)
    ...             for hole in holes
    ...             for vertex in hole
    ...             for coordinate in vertex)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(min_x_coordinate <= vertex_x <= max_x_coordinate
    ...         and min_y_coordinate <= vertex_y <= max_y_coordinate
    ...         for vertex_x, vertex_y in border)
    ...     and all(min_x_coordinate <= vertex_x <= max_x_coordinate
    ...             and min_y_coordinate <= vertex_y <= max_y_coordinate
    ...             for hole in holes
    ...             for vertex_x, vertex_y in hole)
    ...     for border, holes in multipolygon)
    True
    """
    _validate_sizes(min_size, max_size, EMPTY_MULTIPOLYGON_SIZE)
    _validate_sizes(min_border_size, max_border_size, TRIANGULAR_CONTOUR_SIZE,
                    'min_border_size', 'max_border_size')
    _validate_sizes(min_holes_size, max_holes_size, EMPTY_MULTICONTOUR_SIZE,
                    'min_holes_size', 'max_holes_size')
    _validate_sizes(min_hole_size, max_hole_size, TRIANGULAR_CONTOUR_SIZE,
                    'min_hole_size', 'max_hole_size')
    min_size, min_border_size, min_hole_size = (
        max(min_size, EMPTY_MULTIPOLYGON_SIZE),
        max(min_border_size, TRIANGULAR_CONTOUR_SIZE),
        max(min_hole_size, TRIANGULAR_CONTOUR_SIZE))
    if y_coordinates is None:
        y_coordinates = x_coordinates
    min_polygon_points_count = min_border_size + min_holes_size * min_hole_size
    context = get_context()

    @strategies.composite
    def xs_to_multipolygons(draw: Callable[[Strategy[Domain]], Domain],
                            xs: List[Coordinate],
                            edges_constructor: PolygonEdgesConstructor
                            = to_polygon_border_edges_constructor(context)
                            ) -> Multipolygon:
        size_upper_bound = len(xs) // min_polygon_points_count
        size = draw(strategies.integers(min_size,
                                        size_upper_bound
                                        if max_size is None
                                        else min(max_size, size_upper_bound)))
        if not size:
            return []
        xs = sorted(xs)
        multipolygon = []
        start, coordinates_count = 0, len(xs)
        for index in range(size - 1):
            polygon_points_count = draw(strategies.integers(
                    min_polygon_points_count,
                    (coordinates_count - start) // (size - index)))
            polygon_xs = xs[start:start + polygon_points_count]
            polygon = draw(
                    polygons(strategies.sampled_from(polygon_xs),
                             y_coordinates,
                             min_size=min_border_size,
                             max_size=max_border_size,
                             min_holes_size=min_holes_size,
                             max_holes_size=max_holes_size,
                             min_hole_size=min_hole_size,
                             max_hole_size=max_hole_size))
            multipolygon.append(polygon)
            can_touch_next_polygon = not has_vertical_leftmost_segment(
                    edges_constructor(polygon))
            start += polygon_points_count - can_touch_next_polygon
        multipolygon.append(draw(
                polygons(strategies.sampled_from(xs[start:]),
                         y_coordinates,
                         min_size=min_border_size,
                         max_size=max_border_size,
                         min_holes_size=min_holes_size,
                         max_holes_size=max_holes_size,
                         min_hole_size=min_hole_size,
                         max_hole_size=max_hole_size)))
        return multipolygon

    @strategies.composite
    def ys_to_multipolygons(draw: Callable[[Strategy[Domain]], Domain],
                            ys: List[Coordinate],
                            edges_constructor: PolygonEdgesConstructor
                            = to_polygon_border_edges_constructor(context)
                            ) -> Multipolygon:
        size_scale = len(ys) // min_polygon_points_count
        size = draw(strategies.integers(min_size,
                                        size_scale
                                        if max_size is None
                                        else min(max_size, size_scale)))
        if not size:
            return []
        ys = sorted(ys)
        multipolygon = []
        start, coordinates_count = 0, len(ys)
        for index in range(size - 1):
            polygon_points_count = draw(strategies.integers(
                    min_polygon_points_count,
                    (coordinates_count - start) // (size - index)))
            polygon_ys = ys[start:start + polygon_points_count]
            polygon = draw(
                    polygons(x_coordinates,
                             strategies.sampled_from(polygon_ys),
                             min_size=min_border_size,
                             max_size=max_border_size,
                             min_holes_size=min_holes_size,
                             max_holes_size=max_holes_size,
                             min_hole_size=min_hole_size,
                             max_hole_size=max_hole_size))
            multipolygon.append(polygon)
            can_touch_next_polygon = not has_horizontal_lowermost_segment(
                    edges_constructor(polygon))
            start += polygon_points_count - can_touch_next_polygon
        multipolygon.append(draw(
                polygons(x_coordinates,
                         strategies.sampled_from(ys[start:]),
                         min_size=min_border_size,
                         max_size=max_border_size,
                         min_holes_size=min_holes_size,
                         max_holes_size=max_holes_size,
                         min_hole_size=min_hole_size,
                         max_hole_size=max_hole_size)))
        return multipolygon

    min_points_count = min_size * min_polygon_points_count
    max_points_count = (None
                        if (max_size is None
                            or max_border_size is None
                            or max_holes_size is None
                            or max_hole_size is None)
                        else max_size * (max_border_size
                                         + max_hole_size * max_holes_size))

    result = ((strategies.lists(x_coordinates,
                                min_size=min_points_count,
                                max_size=max_points_count,
                                unique=True)
               .flatmap(xs_to_multipolygons))
              | (strategies.lists(y_coordinates,
                                  min_size=min_points_count,
                                  max_size=max_points_count,
                                  unique=True)
                 .flatmap(ys_to_multipolygons)))

    if min_holes_size == EMPTY_MULTICONTOUR_SIZE:
        def multicontour_to_multipolygon(multicontour: Multicontour
                                         ) -> Multipolygon:
            return [(contour, []) for contour in multicontour]

        result = ((multicontours(x_coordinates, y_coordinates,
                                 min_size=min_size,
                                 max_size=max_size,
                                 min_contour_size=min_border_size,
                                 max_contour_size=max_border_size)
                   .map(multicontour_to_multipolygon))
                  | result)
    return result


def mixes(x_coordinates: Strategy[Coordinate],
          y_coordinates: Optional[Strategy[Coordinate]] = None,
          *,
          min_multipoint_size: int = EMPTY_MULTIPOINT_SIZE,
          max_multipoint_size: Optional[int] = None,
          min_multisegment_size: int = EMPTY_MULTISEGMENT_SIZE,
          max_multisegment_size: Optional[int] = None,
          min_multipolygon_size: int = EMPTY_MULTIPOLYGON_SIZE,
          max_multipolygon_size: Optional[int] = None,
          min_multipolygon_border_size: int = TRIANGULAR_CONTOUR_SIZE,
          max_multipolygon_border_size: Optional[int] = None,
          min_multipolygon_holes_size: int = EMPTY_MULTICONTOUR_SIZE,
          max_multipolygon_holes_size: Optional[int] = None,
          min_multipolygon_hole_size: int = TRIANGULAR_CONTOUR_SIZE,
          max_multipolygon_hole_size: Optional[int] = None) -> Strategy[Mix]:
    """
    Returns a strategy for mixes.
    Mix is a triplet of disjoint multipoint, multisegment and multipolygon.

    :param x_coordinates: strategy for vertices' x-coordinates.
    :param y_coordinates:
        strategy for vertices' y-coordinates,
        ``None`` for reusing x-coordinates strategy.
    :param min_multipoint_size: lower bound for multipoint size.
    :param max_multipoint_size:
        upper bound for multipoint size, ``None`` for unbound.
    :param min_multisegment_size: lower bound for multisegment size.
    :param max_multisegment_size:
        upper bound for multisegment size, ``None`` for unbound.
    :param min_multipolygon_size: lower bound for multipolygon size.
    :param max_multipolygon_size:
        upper bound for multipolygon size, ``None`` for unbound.
    :param min_multipolygon_border_size: lower bound for polygons' border size.
    :param max_multipolygon_border_size:
        upper bound for polygons' border size, ``None`` for unbound.
    :param min_multipolygon_holes_size: lower bound for polygons' holes count.
    :param max_multipolygon_holes_size:
        upper bound for polygons' holes count, ``None`` for countless.
    :param min_multipolygon_hole_size: lower bound for hole size.
    :param max_multipolygon_hole_size:
        upper bound for polygons' hole size, ``None`` for unbound.

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Multipoint, Multisegment = (context.multipoint_cls,
    ...                             context.multisegment_cls)

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> min_multipoint_size, max_multipoint_size = 2, 3
    >>> min_multisegment_size, max_multisegment_size = 1, 4
    >>> min_multipolygon_size, max_multipolygon_size = 0, 5
    >>> min_multipolygon_border_size, max_multipolygon_border_size = 5, 10
    >>> min_multipolygon_holes_size, max_multipolygon_holes_size = 1, 4
    >>> min_multipolygon_hole_size, max_multipolygon_hole_size = 3, 5
    >>> mixes = planar.mixes(
    ...         coordinates,
    ...         min_multipoint_size=min_multipoint_size,
    ...         max_multipoint_size=max_multipoint_size,
    ...         min_multisegment_size=min_multisegment_size,
    ...         max_multisegment_size=max_multisegment_size,
    ...         min_multipolygon_size=min_multipolygon_size,
    ...         max_multipolygon_size=max_multipolygon_size,
    ...         min_multipolygon_border_size=min_multipolygon_border_size,
    ...         max_multipolygon_border_size=max_multipolygon_border_size,
    ...         min_multipolygon_holes_size=min_multipolygon_holes_size,
    ...         max_multipolygon_holes_size=max_multipolygon_holes_size,
    ...         min_multipolygon_hole_size=min_multipolygon_hole_size,
    ...         max_multipolygon_hole_size=max_multipolygon_hole_size)
    >>> mix = mixes.example()
    >>> isinstance(mix, tuple)
    True
    >>> len(mix) == 3
    True
    >>> multipoint, multisegment, multipolygon = mix
    >>> isinstance(multipoint, Multipoint)
    True
    >>> min_multipoint_size <= len(multipoint.points) <= max_multipoint_size
    True
    >>> all(isinstance(point.x, coordinates_type)
    ...     and isinstance(point.y, coordinates_type)
    ...     for point in multipoint.points)
    True
    >>> all(min_coordinate <= point.x <= max_coordinate
    ...     and min_coordinate <= point.y <= max_coordinate
    ...     for point in multipoint.points)
    True
    >>> isinstance(multisegment, Multisegment)
    True
    >>> (min_multisegment_size <= len(multisegment.segments)
    ...  <= max_multisegment_size)
    True
    >>> all(isinstance(segment.start.x, coordinates_type)
    ...     and isinstance(segment.start.y, coordinates_type)
    ...     and isinstance(segment.end.x, coordinates_type)
    ...     and isinstance(segment.end.y, coordinates_type)
    ...     for segment in multisegment.segments)
    True
    >>> all(min_coordinate <= segment.start.x <= max_coordinate
    ...     and min_coordinate <= segment.start.y <= max_coordinate
    ...     and min_coordinate <= segment.end.x <= max_coordinate
    ...     and min_coordinate <= segment.end.y <= max_coordinate
    ...     for segment in multisegment.segments)
    True
    >>> isinstance(multipolygon, list)
    True
    >>> min_multipolygon_size <= len(multipolygon) <= max_multipolygon_size
    True
    >>> all(isinstance(polygon, tuple) for polygon in multipolygon)
    True
    >>> all(len(polygon) == 2 for polygon in multipolygon)
    True
    >>> all(isinstance(border, list)
    ...     and isinstance(holes, list)
    ...     and all(isinstance(hole, list) for hole in holes)
    ...     for border, holes in multipolygon)
    True
    >>> all(min_multipolygon_border_size
    ...     <= len(border)
    ...     <= max_multipolygon_border_size
    ...     and (min_multipolygon_holes_size
    ...          <= len(holes)
    ...          <= max_multipolygon_holes_size)
    ...     and all(min_multipolygon_hole_size
    ...             <= len(hole)
    ...             <= max_multipolygon_hole_size
    ...             for hole in holes)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(isinstance(vertex, Point) for vertex in border)
    ...     and all(isinstance(vertex, Point)
    ...             for hole in holes
    ...             for vertex in hole)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(isinstance(vertex.x, coordinates_type)
    ...         and isinstance(vertex.y, coordinates_type)
    ...         for vertex in border)
    ...     and all(isinstance(vertex.x, coordinates_type)
    ...             and isinstance(vertex.y, coordinates_type)
    ...             for hole in holes
    ...             for vertex in hole)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(min_coordinate <= vertex.x <= max_coordinate
    ...         and min_coordinate <= vertex.y <= max_coordinate
    ...         for vertex in border)
    ...     and all(min_coordinate <= vertex.x <= max_coordinate
    ...             and min_coordinate <= vertex.y <= max_coordinate
    ...             for hole in holes
    ...             for vertex in hole)
    ...     for border, holes in multipolygon)
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
    >>> min_multipoint_size, max_multipoint_size = 2, 3
    >>> min_multisegment_size, max_multisegment_size = 1, 4
    >>> min_multipolygon_size, max_multipolygon_size = 0, 5
    >>> min_multipolygon_border_size, max_multipolygon_border_size = 5, 10
    >>> min_multipolygon_holes_size, max_multipolygon_holes_size = 1, 4
    >>> min_multipolygon_hole_size, max_multipolygon_hole_size = 3, 5
    >>> mixes = planar.mixes(
    ...         x_coordinates, y_coordinates,
    ...         min_multipoint_size=min_multipoint_size,
    ...         max_multipoint_size=max_multipoint_size,
    ...         min_multisegment_size=min_multisegment_size,
    ...         max_multisegment_size=max_multisegment_size,
    ...         min_multipolygon_size=min_multipolygon_size,
    ...         max_multipolygon_size=max_multipolygon_size,
    ...         min_multipolygon_border_size=min_multipolygon_border_size,
    ...         max_multipolygon_border_size=max_multipolygon_border_size,
    ...         min_multipolygon_holes_size=min_multipolygon_holes_size,
    ...         max_multipolygon_holes_size=max_multipolygon_holes_size,
    ...         min_multipolygon_hole_size=min_multipolygon_hole_size,
    ...         max_multipolygon_hole_size=max_multipolygon_hole_size)
    >>> mix = mixes.example()
    >>> isinstance(mix, tuple)
    True
    >>> len(mix) == 3
    True
    >>> multipoint, multisegment, multipolygon = mix
    >>> isinstance(multipoint, Multipoint)
    True
    >>> min_multipoint_size <= len(multipoint.points) <= max_multipoint_size
    True
    >>> all(isinstance(point.x, coordinates_type)
    ...     and isinstance(point.y, coordinates_type)
    ...     for point in multipoint.points)
    True
    >>> all(min_x_coordinate <= point.x <= max_x_coordinate
    ...     and min_y_coordinate <= point.y <= max_y_coordinate
    ...     for point in multipoint.points)
    True
    >>> isinstance(multisegment, Multisegment)
    True
    >>> (min_multisegment_size <= len(multisegment.segments)
    ...  <= max_multisegment_size)
    True
    >>> all(isinstance(segment.start.x, coordinates_type)
    ...     and isinstance(segment.start.y, coordinates_type)
    ...     and isinstance(segment.end.x, coordinates_type)
    ...     and isinstance(segment.end.y, coordinates_type)
    ...     for segment in multisegment.segments)
    True
    >>> all(min_x_coordinate <= segment.start.x <= max_x_coordinate
    ...     and min_y_coordinate <= segment.start.y <= max_y_coordinate
    ...     and min_x_coordinate <= segment.end.x <= max_x_coordinate
    ...     and min_y_coordinate <= segment.end.y <= max_y_coordinate
    ...     for segment in multisegment.segments)
    True
    >>> isinstance(multipolygon, list)
    True
    >>> min_multipolygon_size <= len(multipolygon) <= max_multipolygon_size
    True
    >>> all(isinstance(polygon, tuple) for polygon in multipolygon)
    True
    >>> all(len(polygon) == 2 for polygon in multipolygon)
    True
    >>> all(isinstance(border, list)
    ...     and isinstance(holes, list)
    ...     and all(isinstance(hole, list) for hole in holes)
    ...     for border, holes in multipolygon)
    True
    >>> all(min_multipolygon_border_size
    ...     <= len(border)
    ...     <= max_multipolygon_border_size
    ...     and (min_multipolygon_holes_size
    ...          <= len(holes)
    ...          <= max_multipolygon_holes_size)
    ...     and all(min_multipolygon_hole_size
    ...             <= len(hole)
    ...             <= max_multipolygon_hole_size
    ...             for hole in holes)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(isinstance(vertex, Point) for vertex in border)
    ...     and all(isinstance(vertex, Point)
    ...             for hole in holes
    ...             for vertex in hole)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(isinstance(vertex.x, coordinates_type)
    ...         and isinstance(vertex.y, coordinates_type)
    ...         for vertex in border)
    ...     and all(isinstance(vertex.x, coordinates_type)
    ...             and isinstance(vertex.y, coordinates_type)
    ...             for hole in holes
    ...             for vertex in hole)
    ...     for border, holes in multipolygon)
    True
    >>> all(all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...         and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...         for vertex in border)
    ...     and all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...             and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...             for hole in holes
    ...             for vertex in hole)
    ...     for border, holes in multipolygon)
    True
    """
    _validate_sizes(min_multipoint_size, max_multipoint_size,
                    EMPTY_MULTIPOINT_SIZE, 'min_multipoint_size',
                    'max_multipoint_size')
    _validate_sizes(min_multisegment_size, max_multisegment_size,
                    EMPTY_MULTISEGMENT_SIZE, 'min_multisegment_size',
                    'max_multisegment_size')
    _validate_sizes(min_multipolygon_size, max_multipolygon_size,
                    EMPTY_MULTIPOLYGON_SIZE, 'min_multipolygon_size',
                    'max_multipolygon_size')
    _validate_sizes(min_multipolygon_border_size, max_multipolygon_border_size,
                    TRIANGULAR_CONTOUR_SIZE, 'min_multipolygon_border_size',
                    'max_multipolygon_border_size')
    _validate_sizes(min_multipolygon_holes_size, max_multipolygon_holes_size,
                    EMPTY_MULTICONTOUR_SIZE, 'min_multipolygon_holes_size',
                    'max_multipolygon_holes_size')
    _validate_sizes(min_multipolygon_hole_size, max_multipolygon_hole_size,
                    TRIANGULAR_CONTOUR_SIZE, 'min_multipolygon_hole_size',
                    'max_multipolygon_hole_size')
    min_multipoint_size = max(min_multipoint_size, EMPTY_MULTIPOINT_SIZE)
    min_multisegment_size = max(min_multisegment_size, EMPTY_MULTISEGMENT_SIZE)
    min_multipolygon_size = max(min_multipolygon_size, EMPTY_MULTIPOLYGON_SIZE)
    min_multipolygon_border_size = max(min_multipolygon_border_size,
                                       TRIANGULAR_CONTOUR_SIZE)
    min_multipolygon_hole_size = max(min_multipolygon_hole_size,
                                     TRIANGULAR_CONTOUR_SIZE)
    if y_coordinates is None:
        y_coordinates = x_coordinates

    min_polygon_points_count = (min_multipolygon_border_size
                                + min_multipolygon_holes_size
                                * min_multipolygon_hole_size)
    min_multipolygon_points_count = (min_multipolygon_size
                                     * min_polygon_points_count)
    min_multisegment_points_count = 2 * min_multisegment_size
    min_points_count = (min_multipoint_size + min_multisegment_points_count
                        + min_multipolygon_points_count)
    context = get_context()
    multipoint_cls, multisegments_cls = (context.multipoint_cls,
                                         context.multisegment_cls)

    @strategies.composite
    def xs_to_mix(draw: Callable[[Strategy[Domain]], Domain],
                  xs: List[Coordinate],
                  edges_constructor: PolygonEdgesConstructor
                  = to_polygon_border_edges_constructor(context)) -> Mix:
        (multipoint_points_counts, multisegment_points_counts,
         multipolygon_points_counts) = _to_points_counts(draw, len(xs))
        xs = sorted(xs)
        points_sequence, segments_sequence, multipolygon = [], [], []

        def draw_multipoint(points_count: int) -> None:
            points_sequence.extend(draw(_unique_points_sequences(
                    strategies.sampled_from(xs[:points_count]), y_coordinates,
                    min_size=points_count,
                    max_size=points_count)))

        def draw_multisegment(points_count: int) -> None:
            size = points_count // 2
            segments_sequence.extend(draw(
                    _to_non_crossing_non_overlapping_segments_sequences(
                            strategies.sampled_from(xs[:points_count]),
                            y_coordinates,
                            min_size=size,
                            max_size=size)))

        def draw_polygon(points_count: int) -> None:
            multipolygon.append(draw(polygons(
                    strategies.sampled_from(xs[:points_count]), y_coordinates,
                    min_size=min_multipolygon_border_size,
                    max_size=max_multipolygon_border_size,
                    min_holes_size=min_multipolygon_holes_size,
                    max_holes_size=max_multipolygon_holes_size,
                    min_hole_size=min_multipolygon_hole_size,
                    max_hole_size=max_multipolygon_hole_size)))

        drawers_with_points_counts = draw(strategies.permutations(
                tuple(chain(zip(repeat(draw_multipoint),
                                multipoint_points_counts),
                            zip(repeat(draw_multisegment),
                                multisegment_points_counts),
                            zip(repeat(draw_polygon),
                                multipolygon_points_counts)))))
        for index, (drawer, count) in enumerate(drawers_with_points_counts):
            drawer(count)
            can_touch_next_geometry = (
                    drawer is draw_multisegment
                    and index < len(drawers_with_points_counts) - 1
                    and (drawers_with_points_counts[index + 1]
                         is not draw_multipoint)
                    and not has_vertical_leftmost_segment(segments_sequence)
                    or drawer is draw_polygon
                    and index < len(drawers_with_points_counts) - 1
                    and (drawers_with_points_counts[index + 1]
                         is not draw_multipoint)
                    and
                    not has_vertical_leftmost_segment(
                            edges_constructor(multipolygon[-1])))
            xs = xs[count - can_touch_next_geometry:]
        return (multipoint_cls(points_sequence),
                multisegments_cls(segments_sequence), multipolygon)

    @strategies.composite
    def ys_to_mix(draw: Callable[[Strategy[Domain]], Domain],
                  ys: List[Coordinate],
                  edges_constructor: PolygonEdgesConstructor
                  = to_polygon_border_edges_constructor(context)) -> Mix:
        (multipoint_points_counts, multisegment_points_counts,
         multipolygon_points_counts) = _to_points_counts(draw, len(ys))
        ys = sorted(ys)
        points_sequence, segments_sequence, multipolygon = [], [], []

        def draw_multipoint(points_count: int) -> None:
            points_sequence.extend(draw(_unique_points_sequences(
                    x_coordinates, strategies.sampled_from(ys[:points_count]),
                    min_size=points_count,
                    max_size=points_count)))

        def draw_multisegment(points_count: int) -> None:
            size = points_count // 2
            segments_sequence.extend(draw(
                    _to_non_crossing_non_overlapping_segments_sequences(
                            x_coordinates,
                            strategies.sampled_from(ys[:points_count]),
                            min_size=size,
                            max_size=size)))

        def draw_polygon(points_count: int) -> None:
            multipolygon.append(draw(polygons(
                    x_coordinates, strategies.sampled_from(ys[:points_count]),
                    min_size=min_multipolygon_border_size,
                    max_size=max_multipolygon_border_size,
                    min_holes_size=min_multipolygon_holes_size,
                    max_holes_size=max_multipolygon_holes_size,
                    min_hole_size=min_multipolygon_hole_size,
                    max_hole_size=max_multipolygon_hole_size)))

        drawers_with_points_counts = draw(strategies.permutations(
                tuple(chain(zip(repeat(draw_multipoint),
                                multipoint_points_counts),
                            zip(repeat(draw_multisegment),
                                multisegment_points_counts),
                            zip(repeat(draw_polygon),
                                multipolygon_points_counts)))))
        for index, (drawer, count) in enumerate(drawers_with_points_counts):
            drawer(count)
            can_touch_next_geometry = (
                    drawer is draw_multisegment
                    and index < len(drawers_with_points_counts) - 1
                    and (drawers_with_points_counts[index + 1]
                         is not draw_multipoint)
                    and not has_horizontal_lowermost_segment(segments_sequence)
                    or drawer is draw_polygon
                    and index < len(drawers_with_points_counts) - 1
                    and (drawers_with_points_counts[index + 1]
                         is not draw_multipoint)
                    and
                    not has_horizontal_lowermost_segment(
                            edges_constructor(multipolygon[-1])))
            ys = ys[count - can_touch_next_geometry:]
        return (multipoint_cls(points_sequence),
                multisegments_cls(segments_sequence), multipolygon)

    def _to_points_counts(draw: Callable[[Strategy[Domain]], Domain],
                          max_points_count: int
                          ) -> Tuple[List[int], List[int], List[int]]:
        max_multipolygon_points_count = (max_points_count - min_multipoint_size
                                         - min_multisegment_points_count)
        multipolygon_size_upper_bound = (max_multipolygon_points_count
                                         // min_polygon_points_count)
        multipolygon_size = draw(strategies.integers(
                min_multipolygon_size,
                multipolygon_size_upper_bound
                if max_multipolygon_size is None
                else min(multipolygon_size_upper_bound,
                         max_multipolygon_size)))
        multipolygon_points_counts = (
            [draw(polygons_points_counts)
             for polygons_points_counts in repeat(strategies.integers(
                    min_polygon_points_count,
                    max_multipolygon_points_count // multipolygon_size),
                    multipolygon_size)]
            if multipolygon_size
            else [])
        multipolygon_points_count = sum(multipolygon_points_counts)
        multisegment_points_count_upper_bound = (
                max_points_count - multipolygon_points_count
                - min_multipoint_size)
        max_multisegment_points_count = (
            multisegment_points_count_upper_bound
            if max_multisegment_size is None
            else min(multisegment_points_count_upper_bound,
                     2 * max_multisegment_size))
        multisegment_points_count = draw(strategies.sampled_from(
                range(min_multisegment_points_count,
                      max_multisegment_points_count + 1,
                      2)))
        multipoint_size_upper_bound = (max_points_count
                                       - multisegment_points_count
                                       - multipolygon_points_count)
        multipoint_points_count = (multipoint_size_upper_bound
                                   if max_multipoint_size is None
                                   else min(multipoint_size_upper_bound,
                                            max_multipoint_size))
        multisegment_size = multisegment_points_count // 2
        return (_partition(draw, multipoint_points_count),
                [2 * size for size in _partition(draw, multisegment_size)],
                multipolygon_points_counts)

    def _partition(draw: Callable[[Strategy[Domain]], Domain],
                   value: int) -> List[int]:
        assert value >= 0, 'Value should be non-negative.'
        result = []
        while value:
            part = draw(strategies.integers(1, value))
            result.append(part)
            value -= part
        return result

    return ((strategies.lists(x_coordinates,
                              min_size=min_points_count,
                              unique=True)
             .flatmap(xs_to_mix))
            | (strategies.lists(y_coordinates,
                                min_size=min_points_count,
                                unique=True)
               .flatmap(ys_to_mix)))


def _choosers() -> Strategy[Chooser]:
    return (strategies.randoms(use_true_random=True)
            .map(lambda random: random.choice))


def _concave_vertices_sequences(x_coordinates: Strategy[Coordinate],
                                y_coordinates: Optional[Strategy[Coordinate]],
                                *,
                                min_size: int,
                                max_size: Optional[int]
                                ) -> Strategy[Sequence[Point]]:
    _validate_sizes(min_size, max_size, MIN_CONCAVE_CONTOUR_SIZE)
    min_size = max(min_size, MIN_CONCAVE_CONTOUR_SIZE)

    def to_points_with_sizes(points_sequence: Sequence[Point]
                             ) -> Strategy[Tuple[Sequence[Point], int]]:
        sizes = strategies.integers(min_size,
                                    len(points_sequence)
                                    if max_size is None
                                    else min(len(points_sequence), max_size))
        return strategies.tuples(strategies.just(points_sequence), sizes)

    context = get_context()
    return ((_star_vertices_sequences(x_coordinates, y_coordinates,
                                      min_size=min_size,
                                      max_size=max_size)
             | (_unique_points_sequences(x_coordinates, y_coordinates,
                                         min_size=min_size,
                                         max_size=max_size)
                .filter(to_non_collinear_points_detector(context))
                .flatmap(to_points_with_sizes)
                .map(pack(to_vertices_sequence_factory(context)))
                .filter(partial(_has_valid_size,
                                min_size=min_size,
                                max_size=max_size))))
            .filter(to_non_convex_vertices_detector(context)))


def _convex_vertices_sequences(x_coordinates: Strategy[Coordinate],
                               y_coordinates: Optional[Strategy[Coordinate]],
                               *,
                               min_size: int,
                               max_size: Optional[int]
                               ) -> Strategy[Sequence[Point]]:
    _validate_sizes(min_size, max_size, TRIANGULAR_CONTOUR_SIZE)
    if max_size is not None and max_size == TRIANGULAR_CONTOUR_SIZE:
        return _triangular_vertices_sequences(x_coordinates, y_coordinates)
    min_size = max(min_size, TRIANGULAR_CONTOUR_SIZE)
    context = get_context()
    result = (strategies.builds(to_convex_vertices_sequence_factory(context),
                                _unique_points_sequences(x_coordinates,
                                                         y_coordinates,
                                                         min_size=min_size,
                                                         max_size=max_size),
                                strategies.randoms(use_true_random=True))
              .filter(partial(_has_valid_size,
                              min_size=min_size,
                              max_size=max_size)))
    result = (_rectangular_vertices_sequences(x_coordinates, y_coordinates)
              | result
              if min_size <= RECTANGULAR_CONTOUR_SIZE
              else result)
    return (_triangular_vertices_sequences(x_coordinates, y_coordinates)
            | result
            if min_size == TRIANGULAR_CONTOUR_SIZE
            else result)


def _has_valid_size(sized: Sized,
                    *,
                    min_size: int,
                    max_size: Optional[int]) -> bool:
    size = len(sized)
    return min_size <= size and (max_size is None or size <= max_size)


def _rectangular_vertices_sequences(x_coordinates: Strategy[Coordinate],
                                    y_coordinates: Strategy[Coordinate]
                                    ) -> Strategy[Sequence[Point]]:
    context = get_context()

    def to_vertices(box: Box,
                    point_cls: Type[Point] = context.point_cls
                    ) -> Sequence[Point]:
        return [
            point_cls(box.min_x, box.min_y), point_cls(box.max_x, box.min_y),
            point_cls(box.max_x, box.max_y), point_cls(box.min_x, box.max_y)]

    return boxes(x_coordinates, y_coordinates).map(to_vertices)


def _star_vertices_sequences(x_coordinates: Strategy[Coordinate],
                             y_coordinates: Optional[Strategy[Coordinate]],
                             *,
                             min_size: int,
                             max_size: Optional[int]
                             ) -> Strategy[Sequence[Point]]:
    _validate_sizes(min_size, max_size, TRIANGULAR_CONTOUR_SIZE)
    min_size = max(min_size, TRIANGULAR_CONTOUR_SIZE)
    context = get_context()
    return (_unique_points_sequences(x_coordinates, y_coordinates,
                                     min_size=min_size,
                                     max_size=max_size)
            .filter(to_non_collinear_points_detector(context))
            .map(to_star_contour_vertices_factory(context))
            .filter(partial(_has_valid_size,
                            min_size=min_size,
                            max_size=max_size)))


def _triangular_vertices_sequences(x_coordinates: Strategy[Coordinate],
                                   y_coordinates
                                   : Optional[Strategy[Coordinate]]
                                   ) -> Strategy[Sequence[Point]]:
    context = get_context()

    def to_counterclockwise_vertices(vertices_triplet
                                     : Tuple[Point, Point, Point],
                                     orienteer: Orienteer
                                     = context.angle_orientation
                                     ) -> Tuple[Point, Point, Point]:
        return (vertices_triplet
                if orienteer(*vertices_triplet) is Orientation.COUNTERCLOCKWISE
                else vertices_triplet[::-1])

    vertices = points(x_coordinates, y_coordinates)
    return (strategies.tuples(vertices, vertices, vertices)
            .filter(to_strict_vertices_detector(context))
            .map(to_counterclockwise_vertices)
            .map(list))


def _unique_points_sequences(x_coordinates: Strategy[Coordinate],
                             y_coordinates: Optional[Strategy[Coordinate]],
                             *,
                             min_size: int,
                             max_size: Optional[int]
                             ) -> Strategy[Sequence[Point]]:
    return strategies.lists(points(x_coordinates, y_coordinates),
                            unique=True,
                            min_size=min_size,
                            max_size=max_size)


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


def _vertices_sequences(x_coordinates: Strategy[Coordinate],
                        y_coordinates: Optional[Strategy[Coordinate]],
                        *,
                        min_size: int,
                        max_size: Optional[int]) -> Strategy[Sequence[Point]]:
    _validate_sizes(min_size, max_size, TRIANGULAR_CONTOUR_SIZE)
    if max_size is not None and max_size == TRIANGULAR_CONTOUR_SIZE:
        return _triangular_vertices_sequences(x_coordinates, y_coordinates)
    return (_convex_vertices_sequences(x_coordinates, y_coordinates,
                                       min_size=max(min_size,
                                                    TRIANGULAR_CONTOUR_SIZE),
                                       max_size=max_size)
            | _concave_vertices_sequences(x_coordinates, y_coordinates,
                                          min_size=
                                          max(min_size,
                                              MIN_CONCAVE_CONTOUR_SIZE),
                                          max_size=max_size))
