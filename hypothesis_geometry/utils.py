from functools import partial
from math import atan2
from random import Random
from typing import (Callable,
                    Iterable,
                    List,
                    Optional,
                    Sequence,
                    Tuple,
                    TypeVar)

from dendroid import red_black
from dendroid.hints import Sortable
from robust.linear import (SegmentsRelationship,
                           segments_relationship)

from .core import triangular
from .core.subdivisional import (QuadEdge,
                                 to_edge_neighbours)
from .core.utils import (Orientation,
                         orientation)
from .hints import (Contour,
                    Coordinate,
                    Point,
                    Polygon,
                    Segment)


def to_concave_contour(points: Sequence[Point], size: int) -> Contour:
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
    boundary_edges = triangular.to_boundary_edges(triangulation)
    boundary_vertices = {edge.start for edge in boundary_edges}

    def is_mouth(edge: QuadEdge) -> bool:
        return edge.left_from_start.end not in boundary_vertices

    edges_neighbours = {edge: to_edge_neighbours(edge)
                        for edge in boundary_edges}
    candidates = red_black.tree(*filter(is_mouth, boundary_edges),
                                key=_edge_key)
    current_size = len(to_strict_convex_hull(points))
    while current_size < size:
        try:
            edge = candidates.popmax()
        except KeyError:
            break
        if not is_mouth(edge):
            continue
        size += 1
        boundary_vertices.add(edge.left_from_start.end)
        triangulation.delete(edge)
        for neighbour in edges_neighbours.pop(edge):
            edges_neighbours[neighbour] = to_edge_neighbours(neighbour)
            candidates.add(neighbour)
    result = [edge.start
              for edge in triangular.to_boundary_edges(triangulation)]
    shrink_collinear_vertices(result)
    return result


def to_polygon(points: Sequence[Point],
               size: int,
               holes_sizes: List[int]) -> Polygon:
    triangulation = triangular.delaunay(points)
    boundary_edges = triangular.to_boundary_edges(triangulation)
    boundary_vertices = {edge.start for edge in boundary_edges}
    inner_points = sorted(set(points) - boundary_vertices)
    start = 0
    holes = []
    holes_segments = []
    for hole_size in holes_sizes:
        hole_points = inner_points[start:start + hole_size]
        hole = to_concave_contour(hole_points, hole_size)[::-1]
        holes.append(hole)
        holes_segments.extend(contour_to_segments(hole))
        boundary_vertices.update(hole_points)
        start += hole_size

    def is_mouth(edge: QuadEdge) -> bool:
        neighbour_end = edge.left_from_start.end
        return (neighbour_end not in boundary_vertices
                and not any(segments_cross_or_overlap((endpoint,
                                                       neighbour_end),
                                                      hole_segment)
                            for hole_segment in holes_segments
                            for endpoint in (edge.start, edge.end)))

    def segments_cross_or_overlap(left: Segment, right: Segment) -> bool:
        relationship = segments_relationship(left, right)
        return (relationship is SegmentsRelationship.CROSS
                or relationship is SegmentsRelationship.OVERLAP)

    edges_neighbours = {edge: to_edge_neighbours(edge)
                        for edge in boundary_edges}
    candidates = red_black.tree(*filter(is_mouth, boundary_edges),
                                key=_edge_key)
    current_size = len(to_strict_convex_hull(points))
    while current_size < size:
        try:
            edge = candidates.popmax()
        except KeyError:
            break
        if not is_mouth(edge):
            continue
        current_size += 1
        boundary_vertices.add(edge.left_from_start.end)
        triangulation.delete(edge)
        for neighbour in edges_neighbours.pop(edge):
            edges_neighbours[neighbour] = to_edge_neighbours(neighbour)
            candidates.add(neighbour)
    border = [edge.start
              for edge in triangular.to_boundary_edges(triangulation)]
    shrink_collinear_vertices(border)
    return border, holes


def _edge_key(edge: QuadEdge) -> Sortable:
    return _to_squared_edge_length(edge), edge.start, edge.end


def _to_squared_edge_length(edge: QuadEdge) -> Coordinate:
    (start_x, start_y), (end_x, end_y) = edge.start, edge.end
    delta_x, delta_y = start_x - end_x, start_y - end_y
    return delta_x * delta_x + delta_y * delta_y


def constrict_convex_hull_size(points: List[Point],
                               *,
                               max_size: Optional[int]) -> List[Point]:
    if max_size is None:
        return points
    convex_hull = to_strict_convex_hull(points)
    if len(convex_hull) <= max_size:
        return points
    sorted_convex_hull = sorted(
            convex_hull,
            key=partial(_to_squared_points_distance, convex_hull[0]))
    new_border_points = []
    for index in range(max_size):
        quotient, remainder = divmod(index, 2)
        if remainder:
            new_border_points.append(sorted_convex_hull[-quotient - 1])
        else:
            new_border_points.append(sorted_convex_hull[quotient])
    new_border = to_convex_hull(new_border_points)
    new_border_extra_segments = tuple(
            {(new_border[index - 1], new_border[index])
             for index in range(len(new_border))}
            - {(convex_hull[index], convex_hull[index - 1])
               for index in range(len(convex_hull))})
    return (new_border
            + [point
               for point in set(points) - set(convex_hull)
               if all(orientation(end, start, point)
                      is Orientation.COUNTERCLOCKWISE
                      for start, end in new_border_extra_segments)])


def _to_squared_points_distance(left: Point, right: Point) -> Coordinate:
    (left_x, left_y), (right_x, right_y) = left, right
    return (left_x - right_x) ** 2 + (left_y - right_y) ** 2


def to_convex_contour(points: List[Point],
                      random: Random) -> Contour:
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
                               min_coordinate: Coordinate,
                               max_coordinate: Coordinate) -> List[Coordinate]:
        last_min = last_max = min_coordinate
        result = []
        for coordinate in coordinates:
            if random.getrandbits(1):
                result.append(coordinate - last_min)
                last_min = coordinate
            else:
                result.append(last_max - coordinate)
                last_max = coordinate
        result.extend((max_coordinate - last_min, last_max - max_coordinate))
        return result

    vectors_xs = to_vectors_coordinates(xs, min_x, max_x)
    vectors_ys = to_vectors_coordinates(ys, min_y, max_y)
    random.shuffle(vectors_ys)

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
    return to_strict_convex_hull([(min(max(point_x + shift_x, min_x), max_x),
                                   min(max(point_y + shift_y, min_y), max_y))
                                  for point_x, point_y in points])


def shrink_collinear_vertices(contour: Contour) -> None:
    index = -len(contour) + 1
    while index < 0:
        while (max(2, -index) < len(contour)
               and (orientation(contour[index + 2], contour[index + 1],
                                contour[index])
                    is Orientation.COLLINEAR)):
            del contour[index + 1]
        index += 1
    while index < len(contour):
        while (max(2, index) < len(contour)
               and (orientation(contour[index - 2], contour[index - 1],
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
            if (orientation(result[-1], result[-2], point)
                    is Orientation.CLOCKWISE):
                del result[-1]
            else:
                break
        result.append(point)
    return result


def to_strict_convex_hull(points: Sequence[Point]) -> Contour:
    points = sorted(points)
    lower = _to_strict_sub_hull(points)
    upper = _to_strict_sub_hull(reversed(points))
    result = lower[:-1] + upper[:-1]
    shrink_collinear_vertices(result)
    return result


def _to_strict_sub_hull(points: Iterable[Point]) -> List[Point]:
    result = []
    for point in points:
        while len(result) >= 2:
            if (orientation(result[-1], result[-2], point)
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


def sort_pair(pair: Sequence[Domain]) -> Tuple[Domain, Domain]:
    first, second = pair
    return (first, second) if first < second else (second, first)


def contour_to_segments(contour: Contour) -> List[Segment]:
    return [(contour[index - 1], contour[index])
            for index in range(len(contour))]
