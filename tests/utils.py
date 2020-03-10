from itertools import groupby
from numbers import Number
from typing import (Any,
                    Hashable,
                    Iterable,
                    Optional,
                    Tuple,
                    Type,
                    TypeVar)

from hypothesis import strategies

from hypothesis_geometry.hints import (Coordinate,
                                       Point,
                                       Strategy)
from hypothesis_geometry.planar import (TRIANGLE_SIZE,
                                        _has_valid_size)

has_valid_size = _has_valid_size
Domain = TypeVar('Domain')
Limits = Tuple[Coordinate, Optional[Coordinate]]
CoordinatesLimitsType = Tuple[Tuple[Strategy[Coordinate], Limits],
                              Type[Coordinate]]
SizesPair = Tuple[int, Optional[int]]


def identity(argument: Domain) -> Domain:
    return argument


def to_pairs(strategy: Strategy[Domain]) -> Strategy[Tuple[Domain, Domain]]:
    return strategies.tuples(strategy, strategy)


def point_has_valid_size(point: Point) -> bool:
    return len(point) == 2


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


def is_contour(object_: Any) -> bool:
    return (isinstance(object_, list)
            and len(object_) >= TRIANGLE_SIZE
            and all(map(is_point, object_)))


def is_point(object_: Any) -> bool:
    return (isinstance(object_, tuple)
            and len(object_) == 2
            and all(isinstance(coordinate, Number)
                    for coordinate in object_)
            and len(set(map(type, object_))) == 1)
