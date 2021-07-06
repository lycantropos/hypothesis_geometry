from itertools import (accumulate,
                       chain,
                       repeat)
from typing import (Callable,
                    Iterable,
                    List,
                    Optional,
                    Sequence,
                    Type)

from decision.partition import coin_change
from ground.base import (Context,
                         Location,
                         Orientation)
from ground.hints import Point
from reprit.base import generate_repr

from .subdivisional import QuadEdge
from .utils import (ceil_log2,
                    pairwise)


class Triangulation:
    @classmethod
    def delaunay(cls,
                 points: Sequence[Point],
                 context: Context) -> 'Triangulation':
        """Constructs Delaunay triangulation from given points."""
        points = sorted(points)
        result = [cls._initialize_triangulation(points[start:stop],
                                                context)
                  for start, stop in pairwise(accumulate(
                    chain((0,), coin_change(len(points), _base_cases))))]
        for _ in repeat(None, ceil_log2(len(result))):
            parts_to_merge_count = len(result) // 2 * 2
            result = ([result[offset]._merge(result[offset + 1])
                       for offset in range(0, parts_to_merge_count, 2)]
                      + result[parts_to_merge_count:])
        return result[0]

    __slots__ = 'context', 'left_side', 'right_side'

    def __init__(self, left_side: QuadEdge, right_side: QuadEdge,
                 context: Context) -> None:
        self.context, self.left_side, self.right_side = (context, left_side,
                                                         right_side)

    __repr__ = generate_repr(__init__)

    @classmethod
    def from_sides(cls,
                   left_side: QuadEdge,
                   right_side: QuadEdge,
                   context: Context) -> 'Triangulation':
        """Constructs triangulation given its sides."""
        return cls(left_side, right_side, context)

    def delete(self, edge: QuadEdge) -> None:
        """Deletes given edge from the triangulation."""
        if edge is self.right_side or edge.opposite is self.right_side:
            self.right_side = self.right_side.right_from_end.opposite
        if edge is self.left_side or edge.opposite is self.left_side:
            self.left_side = self.left_side.left_from_start
        edge.delete()

    @property
    def _locate_point_in_circle(self) -> Callable[[Point, Point, Point, Point],
                                                  Location]:
        return self.context.locate_point_in_point_point_point_circle

    def _connect(self, base_edge: QuadEdge) -> None:
        while True:
            left_candidate, right_candidate = (
                self._to_left_candidate(base_edge),
                self._to_right_candidate(base_edge))
            if left_candidate is right_candidate is None:
                break
            base_edge = (
                right_candidate.connect(base_edge.opposite)
                if (left_candidate is None
                    or right_candidate is not None
                    and (self._locate_point_in_circle(right_candidate.end,
                                                      left_candidate.end,
                                                      base_edge.end,
                                                      base_edge.start)
                         is Location.INTERIOR))
                else base_edge.opposite.connect(left_candidate.opposite))

    @classmethod
    def _initialize_triangulation(cls,
                                  points: Sequence[Point],
                                  context: Context) -> 'Triangulation':
        return _base_cases[len(points)](cls, points, context)

    def _to_left_candidate(self, base_edge: QuadEdge
                           ) -> Optional[QuadEdge]:
        result = base_edge.opposite.left_from_start
        if (base_edge.orientation_of(result.end)
                is not Orientation.CLOCKWISE):
            return None
        while (self._locate_point_in_circle(result.left_from_start.end,
                                            base_edge.end, base_edge.start,
                                            result.end)
               is Location.INTERIOR
               and (base_edge.orientation_of(result.left_from_start.end)
                    is Orientation.CLOCKWISE)):
            next_candidate = result.left_from_start
            result.delete()
            result = next_candidate
        return result

    def _to_right_candidate(self, base_edge: QuadEdge) -> Optional[QuadEdge]:
        result = base_edge.right_from_start
        if (base_edge.orientation_of(result.end)
                is not Orientation.CLOCKWISE):
            return None
        while (self._locate_point_in_circle(result.right_from_start.end,
                                            base_edge.end, base_edge.start,
                                            result.end)
               is Location.INTERIOR
               and (base_edge.orientation_of(result.right_from_start.end)
                    is Orientation.CLOCKWISE)):
            next_candidate = result.right_from_start
            result.delete()
            result = next_candidate
        return result

    def _find_base_edge(self, other: 'Triangulation') -> QuadEdge:
        while True:
            if (self.right_side.orientation_of(other.left_side.start)
                    is Orientation.COUNTERCLOCKWISE):
                self.right_side = self.right_side.left_from_end
            elif (other.left_side.orientation_of(self.right_side.start)
                  is Orientation.CLOCKWISE):
                other.left_side = other.left_side.right_from_end
            else:
                break
        base_edge = other.left_side.opposite.connect(self.right_side)
        if self.right_side.start == self.left_side.start:
            self.left_side = base_edge.opposite
        if other.left_side.start == other.right_side.start:
            other.right_side = base_edge
        return base_edge

    def _merge(self, other: 'Triangulation') -> 'Triangulation':
        self._connect(self._find_base_edge(other))
        return Triangulation(self.left_side, other.right_side,
                             self.context)


def _triangulate_two_points(cls: Type[Triangulation],
                            sorted_points: Sequence[Point],
                            context: Context) -> Triangulation:
    first_edge = QuadEdge.from_endpoints(*sorted_points,
                                         context=context)
    return cls.from_sides(first_edge, first_edge.opposite, context)


def _triangulate_three_points(cls: Type[Triangulation],
                              sorted_points: Sequence[Point],
                              context: Context) -> Triangulation:
    left_point, mid_point, right_point = sorted_points
    first_edge, second_edge = (QuadEdge.from_endpoints(left_point, mid_point,
                                                       context=context),
                               QuadEdge.from_endpoints(mid_point, right_point,
                                                       context=context))
    first_edge.opposite.splice(second_edge)
    orientation = first_edge.orientation_of(right_point)
    if orientation is Orientation.COUNTERCLOCKWISE:
        second_edge.connect(first_edge)
        return cls.from_sides(first_edge, second_edge.opposite, context)
    elif orientation is Orientation.CLOCKWISE:
        third_edge = second_edge.connect(first_edge)
        return cls.from_sides(third_edge.opposite, third_edge, context)
    else:
        return cls.from_sides(first_edge, second_edge.opposite, context)


_base_cases = {2: _triangulate_two_points,
               3: _triangulate_three_points}

TriangulationBaseConstructor = Callable[[Type[Triangulation], Sequence[Point],
                                         Context],
                                        Triangulation]


def to_boundary_edges(triangulation: Triangulation) -> List[QuadEdge]:
    return list(_to_boundary_edges(triangulation))


def _to_boundary_edges(triangulation: Triangulation) -> Iterable[QuadEdge]:
    # boundary is traversed in counterclockwise direction
    start = triangulation.left_side
    cursor = start
    while True:
        yield cursor
        if cursor.right_from_end is start:
            break
        cursor = cursor.right_from_end
