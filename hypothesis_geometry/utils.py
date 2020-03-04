from functools import partial
from math import atan2
from typing import (Callable,
                    Iterable,
                    List,
                    Sequence,
                    Tuple,
                    TypeVar)

from dendroid import red_black
from dendroid.hints import Sortable

from .core import triangular
from .core.subdivisional import (QuadEdge,
                                 to_edge_neighbours)
from .core.utils import (Orientation,
                         to_orientation)
from .hints import (Contour,
                    Coordinate,
                    Point)


def to_concave_contour(points: Sequence[Point]) -> Contour:
    """
    Based on chi-algorithm by M. Duckham et al.

    Time complexity:
        ``O(len(points) * log len(points))``
    Memory complexity:
        ``O(len(points))``
    Reference:
        http://www.geosensor.net/papers/duckham08.PR.pdf
    """
    triangulation = triangular.delaunay(points)
    boundary = triangular.to_boundary_edges(triangulation)
    boundary_vertices = {edge.start for edge in boundary}

    def is_mouth(edge: QuadEdge) -> bool:
        return edge.left_from_start.end not in boundary_vertices

    edges_neighbours = {edge: to_edge_neighbours(edge)
                        for edge in boundary}

    def edge_key(edge: QuadEdge) -> Sortable:
        return to_squared_edge_length(edge), edge.start, edge.end

    def to_squared_edge_length(edge: QuadEdge) -> Coordinate:
        (start_x, start_y), (end_x, end_y) = edge.start, edge.end
        delta_x, delta_y = start_x - end_x, start_y - end_y
        return delta_x * delta_x + delta_y * delta_y

    candidates = red_black.tree(*filter(is_mouth, boundary),
                                key=edge_key)
    while candidates:
        edge = candidates.popmax()
        if not is_mouth(edge):
            continue
        boundary_vertices.add(edge.left_from_start.end)
        triangulation.delete(edge)
        for neighbour in edges_neighbours.pop(edge):
            edges_neighbours[neighbour] = to_edge_neighbours(neighbour)
            candidates.add(neighbour)
    result = [edge.start for edge in
              triangular.to_boundary_edges(triangulation)]
    shrink_collinear_vertices(result)
    return result


def to_convex_contour(points: List[Point],
                      x_flags: List[bool],
                      y_flags: List[bool],
                      permutation: Sequence[int]) -> Contour:
    """
    Based on Valtr algorithm by Sander Verdonschot.

    Time complexity:
        ``O(len(points) * log len(points))``
    Memory complexity:
        ``O(len(points))``
    Reference:
        http://cglab.ca/~sander/misc/ConvexGeneration/convex.html
    """
    xs, ys = zip(*points)
    xs, ys = sorted(xs), sorted(ys)
    min_x, *xs, max_x = xs
    min_y, *ys, max_y = ys

    def to_vectors_coordinates(coordinates: List[Coordinate],
                               flags: List[bool],
                               min_coordinate: Coordinate,
                               max_coordinate: Coordinate) -> List[Coordinate]:
        last_min = last_max = min_coordinate
        result = []
        for flag, coordinate in zip(flags, coordinates):
            if flag:
                result.append(coordinate - last_min)
                last_min = coordinate
            else:
                result.append(last_max - coordinate)
                last_max = coordinate
        result.extend((max_coordinate - last_min, last_max - max_coordinate))
        return result

    vectors_xs = to_vectors_coordinates(xs, x_flags, min_x, max_x)
    vectors_ys = to_vectors_coordinates(ys, y_flags, min_y, max_y)
    vectors_ys = [vectors_ys[index] for index in permutation]

    def to_vector_angle(vector: Tuple[Coordinate, Coordinate]) -> Sortable:
        x, y = vector
        return atan2(y, x)

    vectors = sorted(zip(vectors_xs, vectors_ys),
                     key=to_vector_angle)
    point_x = point_y = 0
    min_polygon_x = min_polygon_y = 0
    points = []
    for vector_x, vector_y in vectors:
        points.append((point_x, point_y))
        point_x += vector_x
        point_y += vector_y
        min_polygon_x, min_polygon_y = (min(min_polygon_x, point_x),
                                        min(min_polygon_y, point_y))
    shift_x, shift_y = min_x - min_polygon_x, min_y - min_polygon_y
    contour = [(min(max(point_x + shift_x, min_x), max_x),
                min(max(point_y + shift_y, min_y), max_y))
               for point_x, point_y in points]
    shrink_collinear_vertices(contour)
    return to_convex_hull(contour)


def shrink_collinear_vertices(contour: Contour) -> None:
    index = -len(contour) + 1
    while index < 0:
        while (max(2, -index) < len(contour)
               and (to_orientation(contour[index + 2], contour[index + 1],
                                   contour[index])
                    is Orientation.COLLINEAR)):
            del contour[index + 1]
        index += 1
    while index < len(contour):
        while (max(2, index) < len(contour)
               and (to_orientation(contour[index - 2], contour[index - 1],
                                   contour[index])
                    is Orientation.COLLINEAR)):
            del contour[index - 1]
        index += 1


def to_convex_hull(points: Sequence[Point]) -> Contour:
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


Domain = TypeVar('Domain')
Range = TypeVar('Range')


def pack(function: Callable[..., Range]
         ) -> Callable[[Iterable[Domain]], Range]:
    return partial(apply, function)


def apply(function: Callable[..., Range],
          args: Iterable[Domain]) -> Range:
    return function(*args)
