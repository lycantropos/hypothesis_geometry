from itertools import chain
from typing import (Iterator,
                    List,
                    Sequence,
                    TypeVar)

from robust.angular import (Orientation,
                            orientation)

from hypothesis_geometry.hints import Contour

Domain = TypeVar('Domain')
flatten = chain.from_iterable


def split(sequence: Sequence[Domain],
          *,
          size: int = 2) -> List[Sequence[Domain]]:
    step, offset = divmod(len(sequence), size)
    return [sequence[number * step + min(number, offset):
                     (number + 1) * step + min(number + 1, offset)]
            for number in range(size)]


Orientation = Orientation
orientation = orientation


def to_orientations(contour: Contour) -> Iterator[Orientation]:
    return (orientation(contour[index - 1], contour[index],
                        contour[(index + 1) % len(contour)])
            for index in range(len(contour)))
