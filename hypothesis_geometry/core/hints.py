from typing import (Callable as _Callable,
                    Sequence as _Sequence,
                    TypeVar as _TypeVar)

from ground import hints as _hints
from ground.base import Orientation as _Orientation
from hypothesis.strategies import SearchStrategy as _SearchStrategy

Domain = _TypeVar('Domain')
Range = _TypeVar('Range')
Chooser = _Callable[[_Sequence[Domain]], Domain]
Multicontour = _Sequence[_hints.Contour[_hints.Scalar]]
Orienteer = _Callable[[_hints.Point, _hints.Point, _hints.Point], _Orientation]
Strategy = _SearchStrategy
