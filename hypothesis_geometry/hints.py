from numbers import Real
from typing import (List,
                    Tuple,
                    TypeVar)

from ground.hints import (Point,
                          Segment)
from hypothesis.strategies import SearchStrategy

Domain = TypeVar('Domain')
Range = TypeVar('Range')
Strategy = SearchStrategy
Coordinate = Real
Multipoint = List[Point]
Multisegment = List[Segment]
Contour = List[Point]
Polyline = List[Point]
Multicontour = List[Contour]
Polygon = Tuple[Contour, Multicontour]
Multipolygon = List[Polygon]
Mix = Tuple[Multipoint, Multisegment, Multipolygon]
