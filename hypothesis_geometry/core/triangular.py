from itertools import (accumulate,
                       chain)
from typing import (Iterable,
                    List,
                    Optional,
                    Sequence)

from decision.partition import coin_change
from reprit.base import generate_repr

from hypothesis_geometry.hints import Point
from .contracts import is_point_inside_circumcircle
from .subdivisional import QuadEdge
from .utils import (Orientation,
                    orientation,
                    pairwise)


class Triangulation:
    __slots__ = ('left_edge', 'right_edge')

    def __init__(self, left_edge: QuadEdge, right_edge: QuadEdge) -> None:
        self.left_edge = left_edge
        self.right_edge = right_edge

    __repr__ = generate_repr(__init__)

    def merge_with(self, other: 'Triangulation') -> 'Triangulation':
        _merge(self._find_base_edge(other))
        return Triangulation(self.left_edge, other.right_edge)

    def _find_base_edge(self, other: 'Triangulation') -> QuadEdge:
        while True:
            if (self.right_edge.orientation_with(other.left_edge.start)
                    is Orientation.COUNTERCLOCKWISE):
                self.right_edge = self.right_edge.left_from_end
            elif (other.left_edge.orientation_with(self.right_edge.start)
                  is Orientation.CLOCKWISE):
                other.left_edge = other.left_edge.right_from_end
            else:
                break
        base_edge = other.left_edge.opposite.connect(self.right_edge)
        if self.right_edge.start == self.left_edge.start:
            self.left_edge = base_edge.opposite
        if other.left_edge.start == other.right_edge.start:
            other.right_edge = base_edge
        return base_edge

    def delete(self, edge: QuadEdge) -> None:
        if edge is self.right_edge or edge.opposite is self.right_edge:
            self.right_edge = self.right_edge.right_from_end.opposite
        if edge is self.left_edge or edge.opposite is self.left_edge:
            self.left_edge = self.left_edge.left_from_start
        edge.delete()


def delaunay(points: Sequence[Point]) -> Triangulation:
    points = sorted(points)
    result = [_initialize_triangulation(points[start:stop])
              for start, stop in pairwise(accumulate(
                chain((0,), coin_change(len(points), _initializers))))]
    while len(result) > 1:
        parts_to_merge_count = len(result) // 2 * 2
        result = ([result[offset].merge_with(result[offset + 1])
                   for offset in range(0, parts_to_merge_count, 2)]
                  + result[parts_to_merge_count:])
    return result[0]


def _triangulate_two_points(sorted_points: Sequence[Point]) -> Triangulation:
    first_edge = QuadEdge.factory(*sorted_points)
    return Triangulation(first_edge, first_edge.opposite)


def _triangulate_three_points(sorted_points: Sequence[Point]) -> Triangulation:
    left_point, mid_point, right_point = sorted_points
    first_edge, second_edge = (QuadEdge.factory(left_point, mid_point),
                               QuadEdge.factory(mid_point, right_point))
    first_edge.opposite.splice(second_edge)
    angle_orientation = orientation(left_point, mid_point, right_point)
    if angle_orientation is Orientation.COUNTERCLOCKWISE:
        third_edge = second_edge.connect(first_edge)
        return Triangulation(third_edge.opposite, third_edge)
    elif angle_orientation is Orientation.CLOCKWISE:
        second_edge.connect(first_edge)
        return Triangulation(first_edge, second_edge.opposite)
    else:
        return Triangulation(first_edge, second_edge.opposite)


_initializers = {2: _triangulate_two_points,
                 3: _triangulate_three_points}


def _initialize_triangulation(points: Sequence[Point]) -> Triangulation:
    return _initializers[len(points)](points)


def _merge(base_edge: QuadEdge) -> None:
    while True:
        left_candidate, right_candidate = (_to_left_candidate(base_edge),
                                           _to_right_candidate(base_edge))
        if left_candidate is right_candidate is None:
            break
        elif (left_candidate is None
              or right_candidate is not None
              and is_point_inside_circumcircle(left_candidate.end,
                                               base_edge.end,
                                               base_edge.start,
                                               right_candidate.end)):
            base_edge = right_candidate.connect(base_edge.opposite)
        else:
            base_edge = base_edge.opposite.connect(left_candidate.opposite)


def _to_left_candidate(base_edge: QuadEdge) -> Optional[QuadEdge]:
    result = base_edge.opposite.left_from_start
    if base_edge.orientation_with(result.end) is not Orientation.CLOCKWISE:
        return None
    while (is_point_inside_circumcircle(base_edge.end, base_edge.start,
                                        result.end, result.left_from_start.end)
           and (base_edge.orientation_with(result.left_from_start.end)
                is Orientation.CLOCKWISE)):
        next_candidate = result.left_from_start
        result.delete()
        result = next_candidate
    return result


def _to_right_candidate(base_edge: QuadEdge) -> Optional[QuadEdge]:
    result = base_edge.right_from_start
    if base_edge.orientation_with(result.end) is not Orientation.CLOCKWISE:
        return None
    while (is_point_inside_circumcircle(base_edge.end, base_edge.start,
                                        result.end,
                                        result.right_from_start.end)
           and (base_edge.orientation_with(result.right_from_start.end)
                is Orientation.CLOCKWISE)):
        next_candidate = result.right_from_start
        result.delete()
        result = next_candidate
    return result


def to_boundary_edges(triangulation: Triangulation) -> List[QuadEdge]:
    return list(_to_boundary_edges(triangulation))


def _to_boundary_edges(triangulation: Triangulation) -> Iterable[QuadEdge]:
    # boundary is traversed in counterclockwise direction
    start = triangulation.left_edge
    edge = start
    while True:
        yield edge
        if edge.right_from_end is start:
            break
        edge = edge.right_from_end
