from typing import (Callable,
                    Sequence,
                    TypeVar)

from ground.base import Orientation
from ground.hints import Point

Domain = TypeVar('Domain')
Range = TypeVar('Range')
Chooser = Callable[[Sequence[Domain]], Domain]
Orienteer = Callable[[Point, Point, Point], Orientation]
