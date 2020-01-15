from decimal import Decimal
from numbers import Real
from typing import (Sequence,
                    Tuple,
                    TypeVar)

from hypothesis.strategies import SearchStrategy

Strategy = SearchStrategy
Scalar = TypeVar('Scalar', Real, Decimal)
Point = Tuple[Scalar, Scalar]
Contour = Sequence[Point]
