from abc import abstractmethod
from typing import (Callable,
                    MutableSequence,
                    Sequence,
                    Type,
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


class QuadEdge(Protocol):
    """
    Based on:
        quad-edge data structure.

    Reference:
        https://en.wikipedia.org/wiki/Quad-edge
        http://www.sccg.sk/~samuelcik/dgs/quad_edge.pdf
    """

    @classmethod
    @abstractmethod
    def from_endpoints(cls, start: Point, end: Point) -> 'QuadEdge':
        """Creates new edge from endpoints."""

    @property
    def end(self) -> Point:
        """
        aka "Dest" in L. Guibas and J. Stolfi notation.
        """
        return self.opposite.start

    @property
    def left_from_end(self) -> 'QuadEdge':
        """
        aka "Lnext" in L. Guibas and J. Stolfi notation.
        """
        return self.rotated.opposite.left_from_start.rotated

    @property
    @abstractmethod
    def left_from_start(self) -> 'QuadEdge':
        """
        aka "Onext" in L. Guibas and J. Stolfi notation.
        """

    @property
    def opposite(self) -> 'QuadEdge':
        """
        aka "Sym" in L. Guibas and J. Stolfi notation.
        """
        return self.rotated.rotated

    @property
    def right_from_end(self) -> 'QuadEdge':
        """
        aka "Rprev" in L. Guibas and J. Stolfi notation.
        """
        return self.opposite.left_from_start

    @property
    def right_from_start(self) -> 'QuadEdge':
        """
        aka "Oprev" in L. Guibas and J. Stolfi notation.
        """
        return self.rotated.left_from_start.rotated

    @property
    @abstractmethod
    def rotated(self) -> 'QuadEdge':
        """
        aka "Rot" in L. Guibas and J. Stolfi notation.
        """

    @property
    @abstractmethod
    def start(self) -> Point:
        """
        aka "Org" in L. Guibas and J. Stolfi notation.
        """

    @abstractmethod
    def connect(self, other: 'QuadEdge') -> 'QuadEdge':
        """Connects the edge with the other."""

    @abstractmethod
    def delete(self) -> None:
        """Deletes the edge."""

    @abstractmethod
    def orientation_of(self, point: Point) -> Orientation:
        """Returns orientation of the point relative to the edge."""

    @abstractmethod
    def splice(self, other: 'QuadEdge') -> None:
        """Splices the edge with the other."""


class Triangulation(Protocol):
    edge_cls = ...  # type: Type[QuadEdge]

    @classmethod
    @abstractmethod
    def delaunay(cls, points: Sequence[Point]) -> 'Triangulation':
        """Constructs Delaunay triangulation from given points."""

    @classmethod
    @abstractmethod
    def from_sides(cls,
                   left_side: QuadEdge,
                   right_side: QuadEdge) -> 'Triangulation':
        """Constructs triangulation given its sides."""

    left_side = ...  # type: QuadEdge
    right_side = ...  # type: QuadEdge

    @abstractmethod
    def delete(self, edge: QuadEdge) -> None:
        """Deletes given edge from the triangulation."""
