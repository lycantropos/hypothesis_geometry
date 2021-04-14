from typing import (Callable,
                    MutableSequence,
                    Sequence,
                    TypeVar)

from ground.base import Orientation
from ground.hints import (Point,
                          Polygon,
                          Segment)

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

Orienteer = Callable[[Point, Point, Point], Orientation]
Domain = TypeVar('Domain')
Chooser = Callable[[Sequence[Domain]], Domain]
CentroidConstructor = Callable[[Sequence[Point]], Point]
ContourCompressor = Callable[[MutableSequence[Point]], None]
ContourEdgesConstructor = Callable[[Sequence[Point]], Sequence[Segment]]
PointsSequenceOperator = Callable[[Sequence[Point]], Sequence[Point]]
PolygonEdgesConstructor = Callable[[Polygon], Sequence[Segment]]
Range = TypeVar('Range')
QuaternaryPointFunction = Callable[[Point, Point, Point, Point], Range]
