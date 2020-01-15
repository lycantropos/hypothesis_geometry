from decimal import Decimal
from numbers import Real
from typing import (Sequence,
                    Tuple,
                    TypeVar)

from hypothesis.strategies import SearchStrategy

Strategy = SearchStrategy
Coordinate = TypeVar('Coordinate', Real, Decimal)
Point = Tuple[Coordinate, Coordinate]
Contour = Sequence[Point]
