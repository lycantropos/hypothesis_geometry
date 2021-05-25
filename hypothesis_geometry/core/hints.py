from typing import (Callable,
                    Sequence,
                    TypeVar)

from ground.base import Orientation
from ground.hints import (Contour,
                          Point,
                          Scalar)
from hypothesis.strategies import SearchStrategy as Strategy

Domain = TypeVar('Domain')
Range = TypeVar('Range')
Chooser = Callable[[Sequence[Domain]], Domain]
Multicontour = Sequence[Contour[Scalar]]
Orienteer = Callable[[Point, Point, Point], Orientation]
Strategy = Strategy
