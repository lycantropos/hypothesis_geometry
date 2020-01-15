from typing import (Iterable,
                    List,
                    Sequence)

from .core import triangular
from .core.subdivisional import (QuadEdge,
                                 edge_to_endpoints,
                                 edge_to_ring)
from .core.utils import (Orientation,
                         flatten,
                         to_orientation,
                         unique_everseen)
from .hints import (Contour,
                    Point)


def triangulation_to_concave_contour(triangulation: triangular.Triangulation
                                     ) -> Contour:
    boundary = triangulation.to_boundary_edges()

    def is_mouth(edge: QuadEdge) -> bool:
        neighbours = triangulation.to_neighbours(edge)
        return len(neighbours) == 2 and not (neighbours & boundary)

    mouths = {edge: triangulation.to_neighbours(edge)
              for edge in unique_everseen(boundary,
                                          key=edge_to_endpoints)
              if is_mouth(edge)}
    points = set(flatten((edge.start, edge.end)
                         for edge in triangulation.to_edges()))
    for _ in range(len(points) - len(boundary) // 2):
        try:
            edge, neighbours = mouths.popitem()
        except KeyError:
            break
        boundary.remove(edge)
        boundary.remove(edge.opposite)
        triangulation.delete(edge)
        boundary.update(flatten((neighbour, neighbour.opposite)
                                for neighbour in neighbours))
        mouths.update((edge, triangulation.to_neighbours(edge))
                      for edge in neighbours
                      if is_mouth(edge))
        for edge in edge_to_ring(next(iter(neighbours)).opposite):
            mouths.pop(edge.left_from_end, None)
            mouths.pop(edge.left_from_end.opposite, None)
            mouths.pop(edge.right_from_end, None)
            mouths.pop(edge.right_from_end.opposite, None)
    boundary_endpoints = [edge.start
                          for edge in triangulation._to_boundary_edges()]
    return shrink_collinear_vertices(boundary_endpoints)


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
