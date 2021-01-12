from typing import (Sequence,
                    Tuple)

from ground.hints import (Contour,
                          Multipoint,
                          Multisegment,
                          Point)
from hypothesis.strategies import SearchStrategy

Strategy = SearchStrategy
Polyline = Sequence[Point]
Multicontour = Sequence[Contour]
Polygon = Tuple[Contour, Multicontour]
Multipolygon = Sequence[Polygon]
Mix = Tuple[Multipoint, Multisegment, Multipolygon]
