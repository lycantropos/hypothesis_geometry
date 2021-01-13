import warnings as _warnings
from typing import Optional as _Optional

from ground.base import get_context as _get_context
from ground.hints import (Box as _Box,
                          Contour as _Contour,
                          Coordinate as _Coordinate,
                          Multipoint as _Multipoint,
                          Multipolygon as _Multipolygon,
                          Multisegment as _Multisegment,
                          Point as _Point,
                          Polygon as _Polygon,
                          Segment as _Segment)
from hypothesis.errors import HypothesisWarning as _HypothesisWarning

from .core.base import (boxes as _boxes,
                        concave_vertices_sequences
                        as _concave_vertices_sequences,
                        convex_vertices_sequences
                        as _convex_vertices_sequences,
                        mixes as _mixes,
                        multicontours as _multicontours,
                        multipoints as _multipoints,
                        multipolygons as _multipolygons,
                        multisegments as _multisegments,
                        points as _points,
                        polygons as _polygons,
                        rectangular_vertices_sequences
                        as _rectangular_vertices_sequences,
                        segments as _segments,
                        star_vertices_sequences as _star_vertices_sequences,
                        triangular_vertices_sequences
                        as _triangular_vertices_sequences,
                        vertices_sequences as _vertices_sequences)
from .core.constants import (MIN_CONTOUR_SIZE as _MIN_CONTOUR_SIZE,
                             MinContourSize as _MinContourSize)
from .hints import (Mix as _Mix,
                    Multicontour as _Multicontour,
                    Strategy as _Strategy)


def points(x_coordinates: _Strategy[_Coordinate],
           y_coordinates: _Optional[_Strategy[_Coordinate]] = None
           ) -> _Strategy[_Point]:
    """
    Returns a strategy for points.

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
    return _points(x_coordinates, y_coordinates,
                   context=_get_context())


def multipoints(x_coordinates: _Strategy[_Coordinate],
                y_coordinates: _Optional[_Strategy[_Coordinate]] = None,
                *,
                min_size: int = 0,
                max_size: _Optional[int] = None) -> _Strategy[_Multipoint]:
    """
    Returns a strategy for multipoints.

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
    _validate_sizes(min_size, max_size, 0)
    context = _get_context()
    return _multipoints(x_coordinates, y_coordinates,
                        min_size=max(min_size, 0),
                        max_size=max_size,
                        context=context)


def segments(x_coordinates: _Strategy[_Coordinate],
             y_coordinates: _Optional[_Strategy[_Coordinate]] = None
             ) -> _Strategy[_Segment]:
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
    return _segments(x_coordinates, y_coordinates,
                     context=_get_context())


def multisegments(x_coordinates: _Strategy[_Coordinate],
                  y_coordinates: _Optional[_Strategy[_Coordinate]] = None,
                  *,
                  min_size: int = 0,
                  max_size: _Optional[int] = None) -> _Strategy[_Multisegment]:
    """
    Returns a strategy for multisegments.

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
    _validate_sizes(min_size, max_size, 0)
    return _multisegments(x_coordinates, y_coordinates,
                          min_size=max(min_size, 0),
                          max_size=max_size,
                          context=_get_context())


def contours(x_coordinates: _Strategy[_Coordinate],
             y_coordinates: _Optional[_Strategy[_Coordinate]] = None,
             *,
             min_size: int = _MIN_CONTOUR_SIZE,
             max_size: _Optional[int] = None) -> _Strategy[_Contour]:
    """
    Returns a strategy for contours.

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
    >>> contours = planar.contours(coordinates,
    ...                            min_size=min_size,
    ...                            max_size=max_size)
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
    >>> contours = planar.contours(x_coordinates, y_coordinates,
    ...                            min_size=min_size,
    ...                            max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, Contour)
    True
    >>> min_size <= len(contour.vertices) <= max_size
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
    _validate_sizes(min_size, max_size, _MIN_CONTOUR_SIZE)
    context = _get_context()
    return (_vertices_sequences(x_coordinates, y_coordinates,
                                min_size=max(min_size, _MIN_CONTOUR_SIZE),
                                max_size=max_size,
                                context=context)
            .map(context.contour_cls))


def convex_contours(x_coordinates: _Strategy[_Coordinate],
                    y_coordinates: _Optional[_Strategy[_Coordinate]] = None,
                    *,
                    min_size: int = _MinContourSize.CONVEX,
                    max_size: _Optional[int] = None) -> _Strategy[_Contour]:
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
    >>> Contour = context.contour_cls

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
    >>> contours = planar.convex_contours(x_coordinates, y_coordinates,
    ...                                   min_size=min_size,
    ...                                   max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, Contour)
    True
    >>> min_size <= len(contour.vertices) <= max_size
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
    _validate_sizes(min_size, max_size, _MinContourSize.CONVEX)
    context = _get_context()
    return (_convex_vertices_sequences(x_coordinates, y_coordinates,
                                       min_size=max(min_size,
                                                    _MinContourSize.CONVEX),
                                       max_size=max_size,
                                       context=context)
            .map(context.contour_cls))


def concave_contours(x_coordinates: _Strategy[_Coordinate],
                     y_coordinates: _Optional[_Strategy[_Coordinate]] = None,
                     *,
                     min_size: int = _MinContourSize.CONCAVE,
                     max_size: _Optional[int] = None) -> _Strategy[_Contour]:
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
    >>> min_size <= len(contour.vertices) <= max_size
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
    _validate_sizes(min_size, max_size, _MinContourSize.CONCAVE)
    context = _get_context()
    return (_concave_vertices_sequences(x_coordinates, y_coordinates,
                                        min_size=max(min_size,
                                                     _MinContourSize.CONCAVE),
                                        max_size=max_size,
                                        context=context)
            .map(context.contour_cls))


def triangular_contours(x_coordinates: _Strategy[_Coordinate],
                        y_coordinates: _Optional[_Strategy[_Coordinate]] = None
                        ) -> _Strategy[_Contour]:
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
    >>> len(contour.vertices) == 3
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
    context = _get_context()
    return (_triangular_vertices_sequences(x_coordinates, y_coordinates,
                                           context=context)
            .map(context.contour_cls))


def rectangular_contours(x_coordinates: _Strategy[_Coordinate],
                         y_coordinates: _Optional[_Strategy[_Coordinate]]
                         = None) -> _Strategy[_Contour]:
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
    >>> Contour = context.contour_cls

    For same coordinates' domain:

    >>> min_coordinate, max_coordinate = -1., 1.
    >>> coordinates_type = float
    >>> coordinates = strategies.floats(min_coordinate, max_coordinate,
    ...                                 allow_infinity=False,
    ...                                 allow_nan=False)
    >>> contours = planar.rectangular_contours(coordinates)
    >>> contour = contours.example()
    >>> isinstance(contour, Contour)
    True
    >>> len(contour.vertices) == 4
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
    >>> contours = planar.rectangular_contours(x_coordinates, y_coordinates)
    >>> contour = contours.example()
    >>> isinstance(contour, Contour)
    True
    >>> len(contour.vertices) == 4
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
    context = _get_context()
    return (_rectangular_vertices_sequences(x_coordinates, y_coordinates,
                                            context=context)
            .map(context.contour_cls))


def boxes(x_coordinates: _Strategy[_Coordinate],
          y_coordinates: _Optional[_Strategy[_Coordinate]] = None
          ) -> _Strategy[_Box]:
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
    return _boxes(x_coordinates, y_coordinates,
                  context=_get_context())


def star_contours(x_coordinates: _Strategy[_Coordinate],
                  y_coordinates: _Optional[_Strategy[_Coordinate]] = None,
                  *,
                  min_size: int = _MIN_CONTOUR_SIZE,
                  max_size: _Optional[int] = None) -> _Strategy[_Contour]:
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
    >>> Contour = context.contour_cls

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
    >>> contours = planar.star_contours(x_coordinates, y_coordinates,
    ...                                 min_size=min_size,
    ...                                 max_size=max_size)
    >>> contour = contours.example()
    >>> isinstance(contour, Contour)
    True
    >>> min_size <= len(contour.vertices) <= max_size
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
    _validate_sizes(min_size, max_size, _MIN_CONTOUR_SIZE)
    context = _get_context()
    return (_star_vertices_sequences(x_coordinates, y_coordinates,
                                     min_size=max(min_size, _MIN_CONTOUR_SIZE),
                                     max_size=max_size,
                                     context=context)
            .map(context.contour_cls))


def multicontours(x_coordinates: _Strategy[_Coordinate],
                  y_coordinates: _Optional[_Strategy[_Coordinate]] = None,
                  *,
                  min_size: int = 0,
                  max_size: _Optional[int] = None,
                  min_contour_size: int = _MIN_CONTOUR_SIZE,
                  max_contour_size: _Optional[int] = None
                  ) -> _Strategy[_Multicontour]:
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
    >>> Contour = context.contour_cls

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
    >>> all(isinstance(contour, Contour) for contour in multicontour)
    True
    >>> min_size <= len(multicontour) <= max_size
    True
    >>> all(min_contour_size <= len(contour.vertices) <= max_contour_size
    ...     for contour in multicontour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for contour in multicontour
    ...     for vertex in contour.vertices)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
    ...     for contour in multicontour
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
    >>> min_contour_size, max_contour_size = 3, 5
    >>> multicontours = planar.multicontours(x_coordinates, y_coordinates,
    ...                                      min_size=min_size,
    ...                                      max_size=max_size,
    ...                                      min_contour_size=min_contour_size,
    ...                                      max_contour_size=max_contour_size)
    >>> multicontour = multicontours.example()
    >>> isinstance(multicontour, list)
    True
    >>> all(isinstance(contour, Contour) for contour in multicontour)
    True
    >>> min_size <= len(multicontour) <= max_size
    True
    >>> all(min_contour_size <= len(contour.vertices) <= max_contour_size
    ...     for contour in multicontour)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for contour in multicontour
    ...     for vertex in contour.vertices)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for contour in multicontour
    ...     for vertex in contour.vertices)
    True
    """
    _validate_sizes(min_size, max_size, 0)
    _validate_sizes(min_contour_size, max_contour_size, _MIN_CONTOUR_SIZE,
                    'min_contour_size', 'max_contour_size')
    return _multicontours(x_coordinates, y_coordinates,
                          min_size=max(min_size, 0),
                          max_size=max_size,
                          min_contour_size=max(min_contour_size,
                                               _MIN_CONTOUR_SIZE),
                          max_contour_size=max_contour_size,
                          context=_get_context())


def polygons(x_coordinates: _Strategy[_Coordinate],
             y_coordinates: _Optional[_Strategy[_Coordinate]] = None,
             *,
             min_size: int = _MIN_CONTOUR_SIZE,
             max_size: _Optional[int] = None,
             min_holes_size: int = 0,
             max_holes_size: _Optional[int] = None,
             min_hole_size: int = _MIN_CONTOUR_SIZE,
             max_hole_size: _Optional[int] = None) -> _Strategy[_Polygon]:
    """
    Returns a strategy for polygons.

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
    >>> Polygon = context.polygon_cls

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
    >>> isinstance(polygon, Polygon)
    True
    >>> min_size <= len(polygon.border.vertices) <= max_size
    True
    >>> min_holes_size <= len(polygon.holes) <= max_holes_size
    True
    >>> all(min_hole_size <= len(hole.vertices) <= max_hole_size
    ...     for hole in polygon.holes)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in polygon.border.vertices)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for hole in polygon.holes
    ...     for vertex in hole.vertices)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
    ...     for vertex in polygon.border.vertices)
    True
    >>> all(min_coordinate <= vertex.x <= max_coordinate
    ...     and min_coordinate <= vertex.y <= max_coordinate
    ...     for hole in polygon.holes
    ...     for vertex in hole.vertices)
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
    >>> isinstance(polygon, Polygon)
    True
    >>> min_size <= len(polygon.border.vertices) <= max_size
    True
    >>> min_holes_size <= len(polygon.holes) <= max_holes_size
    True
    >>> all(min_hole_size <= len(hole.vertices) <= max_hole_size
    ...     for hole in polygon.holes)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for vertex in polygon.border.vertices)
    True
    >>> all(isinstance(vertex.x, coordinates_type)
    ...     and isinstance(vertex.y, coordinates_type)
    ...     for hole in polygon.holes
    ...     for vertex in hole.vertices)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for vertex in polygon.border.vertices)
    True
    >>> all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...     and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...     for hole in polygon.holes
    ...     for vertex in hole.vertices)
    True
    """
    _validate_sizes(min_size, max_size, _MIN_CONTOUR_SIZE)
    _validate_sizes(min_holes_size, max_holes_size, 0,
                    'min_holes_size', 'max_holes_size')
    _validate_sizes(min_hole_size, max_hole_size, _MIN_CONTOUR_SIZE,
                    'min_hole_size', 'max_hole_size')
    return _polygons(x_coordinates, y_coordinates,
                     min_size=max(min_size, _MIN_CONTOUR_SIZE),
                     max_size=max_size,
                     min_holes_size=max(min_holes_size, 0),
                     max_holes_size=max_holes_size,
                     min_hole_size=max(min_hole_size, _MIN_CONTOUR_SIZE),
                     max_hole_size=max_hole_size,
                     context=_get_context())


def multipolygons(x_coordinates: _Strategy[_Coordinate],
                  y_coordinates: _Optional[_Strategy[_Coordinate]] = None,
                  *,
                  min_size: int = 0,
                  max_size: _Optional[int] = None,
                  min_border_size: int = _MIN_CONTOUR_SIZE,
                  max_border_size: _Optional[int] = None,
                  min_holes_size: int = 0,
                  max_holes_size: _Optional[int] = None,
                  min_hole_size: int = _MIN_CONTOUR_SIZE,
                  max_hole_size: _Optional[int] = None
                  ) -> _Strategy[_Multipolygon]:
    """
    Returns a strategy for multipolygons.

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

    >>> from ground.base import get_context
    >>> from hypothesis import strategies
    >>> from hypothesis_geometry import planar
    >>> context = get_context()
    >>> Multipolygon = context.multipolygon_cls

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
    >>> isinstance(multipolygon, Multipolygon)
    True
    >>> min_size <= len(multipolygon.polygons) <= max_size
    True
    >>> all(min_border_size <= len(polygon.border.vertices) <= max_border_size
    ...     and min_holes_size <= len(polygon.holes) <= max_holes_size
    ...     and all(min_hole_size <= len(hole.vertices) <= max_hole_size
    ...             for hole in polygon.holes)
    ...     for polygon in multipolygon.polygons)
    True
    >>> all(all(isinstance(vertex.x, coordinates_type)
    ...         and isinstance(vertex.y, coordinates_type)
    ...         for vertex in polygon.border.vertices)
    ...     and all(isinstance(vertex.x, coordinates_type)
    ...             and isinstance(vertex.y, coordinates_type)
    ...             for hole in polygon.holes
    ...             for vertex in hole.vertices)
    ...     for polygon in multipolygon.polygons)
    True
    >>> all(all(min_coordinate <= vertex.x <= max_coordinate
    ...         and min_coordinate <= vertex.y <= max_coordinate
    ...         for vertex in polygon.border.vertices)
    ...     and all(min_coordinate <= vertex.x <= max_coordinate
    ...             and min_coordinate <= vertex.y <= max_coordinate
    ...             for hole in polygon.holes
    ...             for vertex in hole.vertices)
    ...     for polygon in multipolygon.polygons)
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
    >>> isinstance(multipolygon, Multipolygon)
    True
    >>> min_size <= len(multipolygon.polygons) <= max_size
    True
    >>> all(min_border_size <= len(polygon.border.vertices) <= max_border_size
    ...     and min_holes_size <= len(polygon.holes) <= max_holes_size
    ...     and all(min_hole_size <= len(hole.vertices) <= max_hole_size
    ...             for hole in polygon.holes)
    ...     for polygon in multipolygon.polygons)
    True
    >>> all(all(isinstance(vertex.x, coordinates_type)
    ...         and isinstance(vertex.y, coordinates_type)
    ...         for vertex in polygon.border.vertices)
    ...     and all(isinstance(vertex.x, coordinates_type)
    ...             and isinstance(vertex.y, coordinates_type)
    ...             for hole in polygon.holes
    ...             for vertex in hole.vertices)
    ...     for polygon in multipolygon.polygons)
    True
    >>> all(all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...         and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...         for vertex in polygon.border.vertices)
    ...     and all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...             and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...             for hole in polygon.holes
    ...             for vertex in hole.vertices)
    ...     for polygon in multipolygon.polygons)
    True
    """
    _validate_sizes(min_size, max_size, 0)
    _validate_sizes(min_border_size, max_border_size, _MIN_CONTOUR_SIZE,
                    'min_border_size', 'max_border_size')
    _validate_sizes(min_holes_size, max_holes_size, 0, 'min_holes_size',
                    'max_holes_size')
    _validate_sizes(min_hole_size, max_hole_size, _MIN_CONTOUR_SIZE,
                    'min_hole_size', 'max_hole_size')
    context = _get_context()
    return _multipolygons(x_coordinates, y_coordinates,
                          min_size=max(min_size, 0),
                          max_size=max_size,
                          min_border_size=max(min_border_size,
                                              _MIN_CONTOUR_SIZE),
                          max_border_size=max_border_size,
                          min_holes_size=min_holes_size,
                          max_holes_size=max_holes_size,
                          min_hole_size=max(min_hole_size, _MIN_CONTOUR_SIZE),
                          max_hole_size=max_hole_size,
                          context=context)


def mixes(x_coordinates: _Strategy[_Coordinate],
          y_coordinates: _Optional[_Strategy[_Coordinate]] = None,
          *,
          min_multipoint_size: int = 0,
          max_multipoint_size: _Optional[int] = None,
          min_multisegment_size: int = 0,
          max_multisegment_size: _Optional[int] = None,
          min_multipolygon_size: int = 0,
          max_multipolygon_size: _Optional[int] = None,
          min_multipolygon_border_size: int = _MIN_CONTOUR_SIZE,
          max_multipolygon_border_size: _Optional[int] = None,
          min_multipolygon_holes_size: int = 0,
          max_multipolygon_holes_size: _Optional[int] = None,
          min_multipolygon_hole_size: int = _MIN_CONTOUR_SIZE,
          max_multipolygon_hole_size: _Optional[int] = None
          ) -> _Strategy[_Mix]:
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
    >>> Multipoint, Multipolygon, Multisegment = (context.multipoint_cls,
    ...                                           context.multipolygon_cls,
    ...                                           context.multisegment_cls)

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
    >>> isinstance(multipolygon, Multipolygon)
    True
    >>> (min_multipolygon_size <= len(multipolygon.polygons)
    ...  <= max_multipolygon_size)
    True
    >>> all(min_multipolygon_border_size
    ...     <= len(polygon.border.vertices)
    ...     <= max_multipolygon_border_size
    ...     and (min_multipolygon_holes_size
    ...          <= len(polygon.holes)
    ...          <= max_multipolygon_holes_size)
    ...     and all(min_multipolygon_hole_size
    ...             <= len(hole.vertices)
    ...             <= max_multipolygon_hole_size
    ...             for hole in polygon.holes)
    ...     for polygon in multipolygon.polygons)
    True
    >>> all(all(isinstance(vertex.x, coordinates_type)
    ...         and isinstance(vertex.y, coordinates_type)
    ...         for vertex in polygon.border.vertices)
    ...     and all(isinstance(vertex.x, coordinates_type)
    ...             and isinstance(vertex.y, coordinates_type)
    ...             for hole in polygon.holes
    ...             for vertex in hole.vertices)
    ...     for polygon in multipolygon.polygons)
    True
    >>> all(all(min_coordinate <= vertex.x <= max_coordinate
    ...         and min_coordinate <= vertex.y <= max_coordinate
    ...         for vertex in polygon.border.vertices)
    ...     and all(min_coordinate <= vertex.x <= max_coordinate
    ...             and min_coordinate <= vertex.y <= max_coordinate
    ...             for hole in polygon.holes
    ...             for vertex in hole.vertices)
    ...     for polygon in multipolygon.polygons)
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
    >>> isinstance(multipolygon, Multipolygon)
    True
    >>> (min_multipolygon_size <= len(multipolygon.polygons)
    ...  <= max_multipolygon_size)
    True
    >>> all(min_multipolygon_border_size
    ...     <= len(polygon.border.vertices)
    ...     <= max_multipolygon_border_size
    ...     and (min_multipolygon_holes_size
    ...          <= len(polygon.holes)
    ...          <= max_multipolygon_holes_size)
    ...     and all(min_multipolygon_hole_size
    ...             <= len(hole.vertices)
    ...             <= max_multipolygon_hole_size
    ...             for hole in polygon.holes)
    ...     for polygon in multipolygon.polygons)
    True
    >>> all(all(isinstance(vertex.x, coordinates_type)
    ...         and isinstance(vertex.y, coordinates_type)
    ...         for vertex in polygon.border.vertices)
    ...     and all(isinstance(vertex.x, coordinates_type)
    ...             and isinstance(vertex.y, coordinates_type)
    ...             for hole in polygon.holes
    ...             for vertex in hole.vertices)
    ...     for polygon in multipolygon.polygons)
    True
    >>> all(all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...         and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...         for vertex in polygon.border.vertices)
    ...     and all(min_x_coordinate <= vertex.x <= max_x_coordinate
    ...             and min_y_coordinate <= vertex.y <= max_y_coordinate
    ...             for hole in polygon.holes
    ...             for vertex in hole.vertices)
    ...     for polygon in multipolygon.polygons)
    True
    """
    _validate_sizes(min_multipoint_size, max_multipoint_size, 0,
                    'min_multipoint_size', 'max_multipoint_size')
    _validate_sizes(min_multisegment_size, max_multisegment_size, 0,
                    'min_multisegment_size', 'max_multisegment_size')
    _validate_sizes(min_multipolygon_size, max_multipolygon_size, 0,
                    'min_multipolygon_size', 'max_multipolygon_size')
    _validate_sizes(min_multipolygon_border_size, max_multipolygon_border_size,
                    _MIN_CONTOUR_SIZE, 'min_multipolygon_border_size',
                    'max_multipolygon_border_size')
    _validate_sizes(min_multipolygon_holes_size, max_multipolygon_holes_size,
                    0, 'min_multipolygon_holes_size',
                    'max_multipolygon_holes_size')
    _validate_sizes(min_multipolygon_hole_size, max_multipolygon_hole_size,
                    _MIN_CONTOUR_SIZE, 'min_multipolygon_hole_size',
                    'max_multipolygon_hole_size')
    context = _get_context()
    return _mixes(
            x_coordinates, y_coordinates,
            min_multipoint_size=max(min_multipoint_size, 0),
            max_multipoint_size=max_multipoint_size,
            min_multisegment_size=max(min_multisegment_size, 0),
            max_multisegment_size=max_multisegment_size,
            min_multipolygon_size=max(min_multipolygon_size, 0),
            max_multipolygon_size=max_multipolygon_size,
            min_multipolygon_border_size=max(min_multipolygon_border_size,
                                             _MIN_CONTOUR_SIZE),
            max_multipolygon_border_size=max_multipolygon_border_size,
            min_multipolygon_holes_size=max(min_multipolygon_holes_size, 0),
            max_multipolygon_holes_size=max_multipolygon_holes_size,
            min_multipolygon_hole_size=max(min_multipolygon_hole_size,
                                           _MIN_CONTOUR_SIZE),
            max_multipolygon_hole_size=max_multipolygon_hole_size,
            context=context)


def _validate_sizes(min_size: int, max_size: _Optional[int],
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
        _warnings.warn('`{min_size_name}` is expected to be '
                       'not less than {min_expected_size}, '
                       'but found {min_size}.'
                       .format(min_size_name=min_size_name,
                               min_expected_size=min_expected_size,
                               min_size=min_size),
                       _HypothesisWarning)
