from decimal import Decimal
from numbers import Real
from typing import (List,
                    Tuple,
                    TypeVar)

from hypothesis.strategies import SearchStrategy

Strategy = SearchStrategy
Coordinate = TypeVar('Coordinate', Real, Decimal)
Point = Tuple[Coordinate, Coordinate]
Segment = Tuple[Point, Point]
Contour = List[Point]
Polyline = List[Point]
Polygon = Tuple[Contour, List[Contour]]
