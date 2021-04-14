from typing import (Callable,
                    Sequence,
                    TypeVar)

from ground.base import Orientation
from ground.hints import Point

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

Orienteer = Callable[[Point, Point, Point], Orientation]
Domain = TypeVar('Domain')
Chooser = Callable[[Sequence[Domain]], Domain]
