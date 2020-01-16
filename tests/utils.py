from typing import (Optional,
                    Tuple,
                    Type)

from hypothesis_geometry.hints import (Coordinate,
                                       Point,
                                       Strategy)
from hypothesis_geometry.planar import _contour_has_valid_size

contour_has_valid_size = _contour_has_valid_size
Limits = Tuple[Coordinate, Optional[Coordinate]]
CoordinatesLimitsType = Tuple[Tuple[Strategy[Coordinate], Limits],
                              Type[Coordinate]]
SizesPair = Tuple[int, Optional[int]]


def point_has_valid_size(point: Point) -> bool:
    return len(point) == 2


def point_has_coordinates_in_range(point: Point,
                                   *,
                                   min_value: Coordinate,
                                   max_value: Optional[Coordinate]) -> bool:
    return all(min_value <= coordinate
               and (max_value is None or coordinate <= max_value)
               for coordinate in point)
