from typing import Sequence

from ground.hints import Contour as _Contour
from hypothesis.strategies import SearchStrategy as Strategy

Strategy = Strategy
Multicontour = Sequence[_Contour]
