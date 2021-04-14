from typing import (Callable,
                    Sequence,
                    TypeVar)

from ground.base import Orientation
from ground.hints import Point

Orienteer = Callable[[Point, Point, Point], Orientation]
Domain = TypeVar('Domain')
Chooser = Callable[[Sequence[Domain]], Domain]
