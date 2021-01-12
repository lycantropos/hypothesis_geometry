from typing import (Sequence,
                    Tuple)

from ground.hints import (Contour,
                          Multipoint,
                          Multisegment,
                          Point,
                          Polygon)
from hypothesis.strategies import SearchStrategy

Strategy = SearchStrategy
Polyline = Sequence[Point]
Multicontour = Sequence[Contour]
Multipolygon = Sequence[Polygon]
Mix = Tuple[Multipoint, Multisegment, Multipolygon]
