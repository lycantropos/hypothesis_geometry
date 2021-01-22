from typing import (Optional,
                    Sequence,
                    Type)

from ground.base import (Context,
                         Orientation)
from ground.hints import Point
from reprit.base import generate_repr

from .hints import QuadEdge


def to_quad_edge_cls(context: Context,
                     base: Type[QuadEdge] = QuadEdge) -> Type[QuadEdge]:
    class Result(base):
        __slots__ = '_start', '_left_from_start', '_rotated'

        def __init__(self,
                     start: Optional[Point] = None,
                     left_from_start: Optional[QuadEdge] = None,
                     rotated: Optional[QuadEdge] = None) -> None:
            self._start, self._left_from_start, self._rotated = (
                start, left_from_start, rotated)

        orientation = staticmethod(context.angle_orientation)

        @property
        def left_from_start(self) -> QuadEdge:
            return self._left_from_start

        @property
        def start(self) -> Point:
            return self._start

        @property
        def rotated(self) -> QuadEdge:
            return self._rotated

        @classmethod
        def from_endpoints(cls, start: Point, end: Point) -> QuadEdge:
            result, opposite = cls(start), cls(end)
            rotated, triple_rotated = cls(), cls()
            result._left_from_start = result
            opposite._left_from_start = opposite
            rotated._left_from_start = triple_rotated
            triple_rotated._left_from_start = rotated
            result._rotated = rotated
            rotated._rotated = opposite
            opposite._rotated = triple_rotated
            triple_rotated._rotated = result
            return result

        __repr__ = generate_repr(from_endpoints)

        def connect(self, other: QuadEdge) -> QuadEdge:
            result = self.from_endpoints(self.end, other.start)
            result.splice(self.left_from_end)
            result.opposite.splice(other)
            return result

        def delete(self) -> None:
            self.splice(self.right_from_start)
            self.opposite.splice(self.opposite.right_from_start)

        def orientation_of(self, point: Point) -> Orientation:
            return self.orientation(self.start, self.end, point)

        def splice(self, other: QuadEdge) -> None:
            alpha = self.left_from_start.rotated
            beta = other.left_from_start.rotated
            self._left_from_start, other._left_from_start = (
                other.left_from_start, self.left_from_start)
            alpha._left_from_start, beta._left_from_start = (
                beta.left_from_start, alpha.left_from_start)

    Result.__name__ = Result.__qualname__ = QuadEdge.__name__
    return Result


def to_edge_neighbours(edge: QuadEdge) -> Sequence[QuadEdge]:
    candidate = edge.left_from_start
    return ((candidate, candidate.right_from_end)
            if (edge.orientation_of(candidate.end)
                is Orientation.COUNTERCLOCKWISE)
            else ())
