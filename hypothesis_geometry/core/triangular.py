from typing import (Iterable,
                    List,
                    Optional,
                    Sequence,
                    Set)

from reprit.base import generate_repr

from hypothesis_geometry.hints import (Contour,
                                       Point)
from .contracts import is_point_inside_circumcircle
from .subdivisional import QuadEdge
from .utils import (Orientation,
                    flatten,
                    split,
                    to_orientation)


class Triangulation:
    __slots__ = ('_left_edge', '_right_edge')

    def __init__(self, left_edge: QuadEdge, right_edge: QuadEdge) -> None:
        self._left_edge = left_edge
        self._right_edge = right_edge

    __repr__ = generate_repr(__init__)

    @property
    def left_edge(self) -> QuadEdge:
        return self._left_edge

    @property
    def right_edge(self) -> QuadEdge:
        return self._right_edge

    def merge_with(self, other: 'Triangulation') -> 'Triangulation':
        _merge(self._find_base_edge(other))
        return Triangulation(self._left_edge, other._right_edge)

    def _find_base_edge(self, other: 'Triangulation') -> QuadEdge:
        while True:
            if (self._right_edge.orientation_with(other._left_edge.start)
                    is Orientation.COUNTERCLOCKWISE):
                self._right_edge = self._right_edge.left_from_end
            elif (other._left_edge.orientation_with(self._right_edge.start)
                  is Orientation.CLOCKWISE):
                other._left_edge = other._left_edge.right_from_end
            else:
                break
        base_edge = other._left_edge.opposite.connect(self._right_edge)
        if self._right_edge.start == self._left_edge.start:
            self._left_edge = base_edge.opposite
        if other._left_edge.start == other._right_edge.start:
            other._right_edge = base_edge
        return base_edge

    def to_triangles(self) -> List[Contour]:
        return list(self._to_triangles())

    def _to_triangles(self) -> Iterable[Contour]:
        visited_vertices = set()
        edges = self.to_edges()
        edges_endpoints = {frozenset((edge.start, edge.end)) for edge in edges}
        for edge in edges:
            if (edge.orientation_with(edge.left_from_start.end)
                    is Orientation.COUNTERCLOCKWISE):
                contour = (edge.start, edge.end, edge.left_from_start.end)
                vertices = frozenset(contour)
                if vertices not in visited_vertices:
                    if (frozenset((edge.end, edge.left_from_start.end))
                            not in edges_endpoints):
                        continue
                    visited_vertices.add(vertices)
                    yield contour

    @staticmethod
    def to_non_adjacent_vertices(edge: QuadEdge) -> Set[Point]:
        return {neighbour.end
                for neighbour in Triangulation._to_incidents(edge)}

    def to_edges(self) -> Set[QuadEdge]:
        result = {self.right_edge, self.left_edge}
        queue = [self.right_edge.left_from_start,
                 self.right_edge.left_from_end,
                 self.right_edge.right_from_start,
                 self.right_edge.right_from_end]
        while queue:
            edge = queue.pop()
            if edge not in result:
                result.update((edge, edge.opposite))
                queue.extend((edge.left_from_start, edge.left_from_end,
                              edge.right_from_start, edge.right_from_end))
        return result

    def to_boundary_edges(self) -> Set[QuadEdge]:
        return set(flatten((edge, edge.opposite)
                           for edge in self._to_boundary_edges()))

    def _to_boundary_edges(self) -> Iterable[QuadEdge]:
        start = self.left_edge
        edge = start
        while True:
            yield edge
            if edge.right_from_end is start:
                break
            edge = edge.right_from_end

    def to_inner_edges(self) -> Set[QuadEdge]:
        return self.to_edges() - self.to_boundary_edges()

    @staticmethod
    def to_neighbours(edge: QuadEdge) -> Set[QuadEdge]:
        return set(Triangulation._to_neighbours(edge))

    @staticmethod
    def _to_neighbours(edge: QuadEdge) -> Iterable[QuadEdge]:
        yield from Triangulation._to_incidents(edge)
        yield from Triangulation._to_incidents(edge.opposite)

    @staticmethod
    def _to_incidents(edge: QuadEdge) -> Iterable[QuadEdge]:
        if (edge.orientation_with(edge.right_from_start.end)
                is Orientation.CLOCKWISE):
            yield edge.right_from_start
        if (edge.orientation_with(edge.left_from_start.end)
                is Orientation.COUNTERCLOCKWISE):
            yield edge.left_from_start

    def delete(self, edge: QuadEdge) -> None:
        if edge is self._right_edge or edge.opposite is self._right_edge:
            self._right_edge = self._right_edge.right_from_end.opposite
        if edge is self._left_edge or edge.opposite is self._left_edge:
            self._left_edge = self._left_edge.left_from_start
        edge.delete()


def delaunay(points: Sequence[Point]) -> Triangulation:
    result = [tuple(sorted(points))]
    while max(map(len, result)) > max(_initializers):
        result = list(flatten(split(part) if len(part) > max(_initializers)
                              else [part]
                              for part in result))
    result = [_initialize_triangulation(points) for points in result]
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
    orientation = to_orientation(left_point, mid_point, right_point)
    if orientation is Orientation.COUNTERCLOCKWISE:
        third_edge = second_edge.connect(first_edge)
        return Triangulation(third_edge.opposite, third_edge)
    elif orientation is Orientation.CLOCKWISE:
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
        left_candidate = _to_left_candidate(base_edge)
        right_candidate = _to_right_candidate(base_edge)
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
