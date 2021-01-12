from typing import (Sequence,
                    Tuple)

from ground.hints import (Contour,
                          Multipoint,
                          Multisegment,
                          Polygon)
from hypothesis.strategies import SearchStrategy

Strategy = SearchStrategy
Multicontour = Sequence[Contour]
Multipolygon = Sequence[Polygon]
Mix = Tuple[Multipoint, Multisegment, Multipolygon]
