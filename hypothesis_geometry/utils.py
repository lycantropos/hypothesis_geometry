from typing import (Iterable,
                    List,
                    Sequence,
                    Set)

from .core import triangular
from .core.subdivisional import QuadEdge
from .core.utils import (Orientation,
                         to_orientation)
from .hints import (Contour,
                    Point)


def triangulation_to_concave_contour(triangulation: triangular.Triangulation
                                     ) -> Contour:
    boundary = to_triangulation_boundary(triangulation)
    boundary_vertices = {edge.start for edge in boundary}

    def is_mouth(edge: QuadEdge) -> bool:
        return edge.left_from_start.end not in boundary_vertices

    candidates = {edge: to_edge_neighbours(edge)
                  for edge in boundary
                  if is_mouth(edge)}
    while candidates:
        edge, neighbours = candidates.popitem()
        if not is_mouth(edge):
            continue
        boundary_vertices.add(edge.left_from_start.end)
        triangulation.delete(edge)
        candidates.update((edge, to_edge_neighbours(edge))
                          for edge in neighbours)
    boundary_endpoints = [edge.start
                          for edge in to_triangulation_boundary(triangulation)]
    return shrink_collinear_vertices(boundary_endpoints)


def to_triangulation_boundary(triangulation: triangular.Triangulation
                              ) -> List[QuadEdge]:
    return list(_to_triangulation_boundary(triangulation))


def _to_triangulation_boundary(triangulation: triangular.Triangulation
                               ) -> Iterable[QuadEdge]:
    # boundary is traversed in counterclockwise direction
    start = triangulation.left_edge
    edge = start
    while True:
        yield edge
        if edge.right_from_end is start:
            break
        edge = edge.right_from_end


def to_edge_neighbours(edge: QuadEdge) -> Set[QuadEdge]:
    return set(_to_edge_neighbours(edge))


def _to_edge_neighbours(edge: QuadEdge) -> Iterable[QuadEdge]:
    candidate = edge.left_from_start
    if (edge.orientation_with(candidate.end)
            is Orientation.COUNTERCLOCKWISE):
        yield candidate
        yield candidate.right_from_end


def shrink_collinear_vertices(contour: Contour) -> Contour:
    result = [contour[0], contour[1]]
    for vertex in contour[2:]:
        while (len(result) > 2
               and (to_orientation(result[-2], result[-1], vertex)
                    is Orientation.COLLINEAR)):
            del result[-1]
        result.append(vertex)
    for index in range(len(result)):
        while (max(index, 2) < len(result)
               and (to_orientation(result[index - 2], result[index - 1],
                                   result[index])
                    is Orientation.COLLINEAR)):
            del result[index - 1]
    return result


def to_convex_hull(points: Sequence[Point]) -> List[Point]:
    points = sorted(points)
    lower = _to_sub_hull(points)
    upper = _to_sub_hull(reversed(points))
    return lower[:-1] + upper[:-1]


def _to_sub_hull(points: Iterable[Point]) -> List[Point]:
    result = []
    for point in points:
        while len(result) >= 2:
            if (to_orientation(result[-1], result[-2], point)
                    is not Orientation.COUNTERCLOCKWISE):
                del result[-1]
            else:
                break
        result.append(point)
    return result
