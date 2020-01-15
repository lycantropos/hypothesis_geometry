from itertools import chain
from numbers import Real
from typing import (Callable,
                    Hashable,
                    Iterable,
                    List,
                    Sequence,
                    TypeVar)

from robust.hints import Point as RealPoint

from hypothesis_geometry.hints import (Point,
                                       Scalar)

Domain = TypeVar('Domain')

flatten = chain.from_iterable


def unique_everseen(iterable: Iterable[Domain],
                    *,
                    key: Callable[[Domain], Hashable] = None
                    ) -> Iterable[Domain]:
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in iterable:
            if element not in seen:
                seen_add(element)
                yield element
    else:
        for element in iterable:
            value = key(element)
            if value not in seen:
                seen_add(value)
                yield element


def split(sequence: Sequence[Domain],
          *,
          size: int = 2) -> List[Sequence[Domain]]:
    step, offset = divmod(len(sequence), size)
    return [sequence[number * step + min(number, offset):
                     (number + 1) * step + min(number + 1, offset)]
            for number in range(size)]


def to_real_point(point: Point) -> RealPoint:
    x, y = point
    return _scalar_to_real(x), _scalar_to_real(y)


def _scalar_to_real(scalar: Scalar) -> Real:
    return scalar if isinstance(scalar, Real) else float(scalar)
