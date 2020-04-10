from numbers import Real
from typing import (List,
                    Tuple)

from hypothesis.strategies import SearchStrategy

Strategy = SearchStrategy
Coordinate = Real
BoundingBox = Tuple[Coordinate, Coordinate, Coordinate, Coordinate]
Point = Tuple[Coordinate, Coordinate]
Segment = Tuple[Point, Point]
Contour = List[Point]
Polyline = List[Point]
Multicontour = List[Contour]
Polygon = Tuple[Contour, Multicontour]
