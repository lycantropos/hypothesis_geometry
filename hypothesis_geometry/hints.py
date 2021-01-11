from typing import (List,
                    Tuple,
                    TypeVar)

from ground.hints import (Multipoint,
                          Multisegment,
                          Point)
from hypothesis.strategies import SearchStrategy

Domain = TypeVar('Domain')
Range = TypeVar('Range')
Strategy = SearchStrategy
Contour = List[Point]
Polyline = List[Point]
Multicontour = List[Contour]
Polygon = Tuple[Contour, Multicontour]
Multipolygon = List[Polygon]
Mix = Tuple[Multipoint, Multisegment, Multipolygon]
