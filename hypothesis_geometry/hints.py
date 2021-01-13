from typing import (Sequence,
                    Tuple)

from ground.hints import (Contour as _Contour,
                          Multipoint as _Multipoint,
                          Multipolygon as _Multipolygon,
                          Multisegment as _Multisegment)
from hypothesis.strategies import SearchStrategy as Strategy

Strategy = Strategy
Multicontour = Sequence[_Contour]
Mix = Tuple[_Multipoint, _Multisegment, _Multipolygon]
