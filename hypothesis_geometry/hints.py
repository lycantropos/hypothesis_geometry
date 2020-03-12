from numbers import Real
from typing import (List,
                    Tuple)

from hypothesis.strategies import SearchStrategy

Strategy = SearchStrategy
Coordinate = Real
Point = Tuple[Coordinate, Coordinate]
Segment = Tuple[Point, Point]
BoundingBox = Tuple[Coordinate, Coordinate, Coordinate, Coordinate]
Contour = List[Point]
Polyline = List[Point]
Polygon = Tuple[Contour, List[Contour]]
