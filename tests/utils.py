from itertools import groupby
from numbers import Number
from typing import (Any,
                    Callable,
                    Hashable,
                    Iterable,
                    List,
                    Optional,
                    Tuple,
                    Type,
                    TypeVar)

from bentley_ottmann.planar import (edges_intersect,
                                    segments_cross_or_overlap)
from hypothesis import strategies
from robust.angular import (Orientation,
                            orientation)

from hypothesis_geometry.hints import (Contour,
                                       Coordinate,
                                       Point,
                                       Strategy)
from hypothesis_geometry.planar import (MIN_POLYLINE_SIZE,
                                        TRIANGULAR_CONTOUR_SIZE,
                                        _has_valid_size)
from hypothesis_geometry.utils import contour_to_segments

has_valid_size = _has_valid_size
Domain = TypeVar('Domain')
Key = Callable[[Domain], Any]
Limits = Tuple[Coordinate, Optional[Coordinate]]
CoordinatesLimitsType = Tuple[Tuple[Strategy[Coordinate], Limits],
                              Type[Coordinate]]
SizesPair = Tuple[int, Optional[int]]


def identity(argument: Domain) -> Domain:
    return argument


def to_pairs(strategy: Strategy[Domain]) -> Strategy[Tuple[Domain, Domain]]:
    return strategies.tuples(strategy, strategy)


def point_has_coordinates_in_range(point: Point,
                                   *,
                                   min_x_value: Coordinate,
                                   max_x_value: Optional[Coordinate],
                                   min_y_value: Coordinate,
                                   max_y_value: Optional[Coordinate]) -> bool:
    x, y = point
    return (is_coordinate_in_range(x,
                                   min_value=min_x_value,
                                   max_value=max_x_value)
            and is_coordinate_in_range(y,
                                       min_value=min_y_value,
                                       max_value=max_y_value))


def is_coordinate_in_range(coordinate: Coordinate,
                           *,
                           min_value: Coordinate,
                           max_value: Optional[Coordinate]) -> bool:
    return (min_value <= coordinate
            and (max_value is None or coordinate <= max_value))


def point_has_coordinates_types(point: Point,
                                *,
                                x_type: Type[Coordinate],
                                y_type: Type[Coordinate]) -> bool:
    x, y = point
    return isinstance(x, x_type) and isinstance(y, y_type)


def has_no_consecutive_repetitions(iterable: Iterable[Domain]) -> bool:
    return any(capacity(group) == 1
               for _, group in groupby(iterable))


def is_counterclockwise_contour(contour: Contour) -> bool:
    index_min = to_index_min(contour)
    return (orientation(contour[index_min - 1], contour[index_min],
                        contour[(index_min + 1) % len(contour)])
            is Orientation.CLOCKWISE)


_sentinel = object()


def to_index_min(values: Iterable[Domain],
                 *,
                 key: Optional[Key] = None,
                 default: Any = _sentinel) -> int:
    kwargs = {}
    if key is not None:
        kwargs['key'] = lambda value_with_index: key(value_with_index[0])
    if default is not _sentinel:
        kwargs['default'] = default
    return min(((value, index)
                for index, value in enumerate(values)),
               **kwargs)[1]


def capacity(iterable: Iterable[Domain]) -> int:
    return sum(1 for _ in iterable)


def all_unique(iterable: Iterable[Hashable]) -> bool:
    seen = set()
    seen_add = seen.add
    for element in iterable:
        if element in seen:
            return False
        seen_add(element)
    return True


def is_polygon(object_: Any) -> bool:
    return (isinstance(object_, tuple)
            and len(object_) == 2
            and is_contour(object_[0])
            and is_multicontour(object_[1]))


def is_multicontour(object_: Any) -> bool:
    return isinstance(object_, list) and all(map(is_contour, object_))


def is_contour(object_: Any) -> bool:
    return (isinstance(object_, list)
            and len(object_) >= TRIANGULAR_CONTOUR_SIZE
            and all(map(is_point, object_)))


def is_polyline(object_: Any) -> bool:
    return (isinstance(object_, list)
            and len(object_) >= MIN_POLYLINE_SIZE
            and all(map(is_point, object_)))


def is_bounding_box(object_: Any) -> bool:
    return (isinstance(object_, tuple)
            and len(object_) == 4
            and all(isinstance(coordinate, Number)
                    for coordinate in object_)
            and len(set(map(type, object_))) == 1)


def is_segment(object_: Any) -> bool:
    return (isinstance(object_, tuple)
            and len(object_) == 2
            and all(map(is_point, object_))
            and len(set(object_)) == 2)


def is_point(object_: Any) -> bool:
    return (isinstance(object_, tuple)
            and len(object_) == 2
            and all(isinstance(coordinate, Number)
                    for coordinate in object_)
            and len(set(map(type, object_))) == 1)


def is_non_self_intersecting_contour(contour: Contour) -> bool:
    return not edges_intersect(contour,
                               accurate=False)


def contours_do_not_cross_or_overlap(contours: List[Contour]) -> bool:
    return not segments_cross_or_overlap(sum([contour_to_segments(contour)
                                              for contour in contours],
                                             []))
