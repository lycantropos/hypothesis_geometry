from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Protocol, TypeAlias, TypeVar

from ground.enums import Orientation
from ground.hints import Contour, Point, Scalar as _Scalar
from typing_extensions import Self


class HasCustomRepr(Protocol):
    @abstractmethod
    def __repr__(self, /) -> str:
        raise NotImplementedError


class Scalar(_Scalar, Protocol):
    def __abs__(self, /) -> Self: ...
    def __float__(self, /) -> float: ...
    def __floor__(self, /) -> int: ...


Domain = TypeVar('Domain')
Range = TypeVar('Range')
Chooser: TypeAlias = Callable[[Sequence[Domain]], Domain]
ScalarT = TypeVar('ScalarT', bound=Scalar)
Multicontour: TypeAlias = Sequence[Contour[ScalarT]]
Orienteer: TypeAlias = Callable[
    [Point[ScalarT], Point[ScalarT], Point[ScalarT]], Orientation
]
