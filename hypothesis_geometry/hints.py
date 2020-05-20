from numbers import Real
from typing import (List,
                    Tuple,
                    TypeVar)

from hypothesis.strategies import SearchStrategy

Domain = TypeVar('Domain')
Range = TypeVar('Range')
Strategy = SearchStrategy
Coordinate = Real
BoundingBox = Tuple[Coordinate, Coordinate, Coordinate, Coordinate]
Point = Tuple[Coordinate, Coordinate]
Segment = Tuple[Point, Point]
Multisegment = List[Segment]
Contour = List[Point]
Polyline = List[Point]
Multicontour = List[Contour]
Polygon = Tuple[Contour, Multicontour]
