import math
from collections import deque
from functools import partial
from itertools import (cycle,
                       groupby)
from math import atan2
from numbers import Real
from operator import (attrgetter,
                      itemgetter)
from random import Random
from typing import (Callable,
                    Collection,
                    Iterable,
                    List,
                    MutableSequence,
                    Optional,
                    Sequence,
                    Tuple,
                    Type)

from dendroid import red_black
from dendroid.hints import (Key,
                            Set)
from ground.base import (Context,
                         Orientation,
                         Relation)
from ground.hints import (Point,
                          Polygon,
                          Scalar,
                          Segment)
from locus import segmental

from .contracts import (angle_contains_point,
                        has_horizontal_lowermost_segment,
                        has_vertical_leftmost_segment)
from .hints import (Chooser,
                    Multicontour,
                    Orienteer)
from .subdivisional import (QuadEdge,
                            to_edge_neighbours)
from .triangular import (Triangulation,
                         to_boundary_edges)


def constrict_convex_hull_size(points: Sequence[Point[Scalar]],
                               *,
                               context: Context,
                               max_size: Optional[int]
                               ) -> Sequence[Point[Scalar]]:
    if max_size is None:
        return points
    convex_hull = context.points_convex_hull(points)
    if len(convex_hull) <= max_size:
        return points
    sorted_convex_hull = sorted(
            convex_hull,
            key=partial(context.points_squared_distance, convex_hull[0]))
    new_border_points = []
    for index in range(max_size):
        quotient, remainder = divmod(index, 2)
        new_border_points.append(sorted_convex_hull[-quotient - 1]
                                 if remainder
                                 else sorted_convex_hull[quotient])
    orienteer = context.angle_orientation
    new_border = list(to_max_convex_hull(new_border_points, orienteer))
    new_border_extra_endpoints_pairs = tuple(
            {(new_border[index - 1], new_border[index])
             for index in range(len(new_border))}
            - {(convex_hull[index], convex_hull[index - 1])
               for index in range(len(convex_hull))})
    return (new_border
            + [point
               for point in set(points) - set(convex_hull)
               if all(orienteer(start, end, point)
                      is Orientation.COUNTERCLOCKWISE
                      for start, end in new_border_extra_endpoints_pairs)])


def _edge_key(edge: QuadEdge) -> Key:
    weight = 0
    cursor = edge
    while True:
        weight += _is_convex_quadrilateral_diagonal(cursor.left_from_end)
        cursor = cursor.left_from_start
        if cursor is edge:
            break
    return (weight, edge.start, edge.end)


def to_multicontour(points: Sequence[Point[Scalar]],
                    sizes: Sequence[int],
                    chooser: Chooser,
                    context: Context) -> Multicontour[Scalar]:
    sorting_key_chooser = partial(chooser, [attrgetter('y', 'x'),
                                            attrgetter('x', 'y')])
    current_sorting_key = sorting_key_chooser()
    points = sorted(points,
                    key=current_sorting_key)
    predicates = cycle((has_vertical_leftmost_segment,
                        has_horizontal_lowermost_segment)
                       if current_sorting_key is None
                       else (has_horizontal_lowermost_segment,
                             has_vertical_leftmost_segment))
    current_predicate = next(predicates)
    contour_cls, to_contour_segments = (context.contour_cls,
                                        context.contour_segments)
    result = []
    for size in sizes:
        contour = contour_cls(to_vertices_sequence(points[:size], size,
                                                   context))
        result.append(contour)
        can_touch_next_contour = current_predicate(
                to_contour_segments(contour))
        points = points[size - can_touch_next_contour:]
        new_sorting_key = sorting_key_chooser()
        if new_sorting_key is not current_sorting_key:
            points.sort(key=new_sorting_key)
            current_sorting_key, current_predicate = (new_sorting_key,
                                                      next(predicates))
    return result


def to_polygon(points: Sequence[Point[Scalar]],
               border_size: int,
               holes_sizes: List[int],
               chooser: Chooser,
               context: Context) -> Polygon[Scalar]:
    triangulation = Triangulation.delaunay(points, context)
    boundary_edges = to_boundary_edges(triangulation)
    boundary_points = {edge.start for edge in boundary_edges}
    sorting_key_chooser = partial(chooser, [attrgetter('y', 'x'),
                                            attrgetter('x', 'y')])
    current_sorting_key = sorting_key_chooser()
    inner_points = sorted(set(points) - boundary_points,
                          key=current_sorting_key)
    predicates = cycle((has_vertical_leftmost_segment,
                        has_horizontal_lowermost_segment)
                       if current_sorting_key is None
                       else (has_horizontal_lowermost_segment,
                             has_vertical_leftmost_segment))
    current_predicate = next(predicates)
    holes, holes_edges = [], []
    contour_cls, to_contour_segments = (context.contour_cls,
                                        context.contour_segments)
    for hole_size in holes_sizes:
        hole_points = inner_points[:hole_size]
        hole = contour_cls(_reverse_vertices(
                to_vertices_sequence(hole_points, hole_size, context)))
        holes.append(hole)
        hole_edges = to_contour_segments(hole)
        holes_edges.extend(hole_edges)
        boundary_points.update(hole_points)
        can_touch_next_hole = current_predicate(hole_edges)
        inner_points = inner_points[hole_size - can_touch_next_hole:]
        next_sorting_key = sorting_key_chooser()
        if next_sorting_key is not current_sorting_key:
            (current_sorting_key, inner_points,
             current_predicate) = (next_sorting_key,
                                   sorted(inner_points,
                                          key=next_sorting_key),
                                   next(predicates))

    def to_edges_cross_or_overlap_detector(edges: Sequence[Segment]
                                           ) -> Callable[[Segment], bool]:
        return ((lambda edge, to_nearest_edge=(segmental.Tree(edges,
                                                              context=context)
                                               .nearest_segment)
                 : segments_cross_or_overlap(to_nearest_edge(edge), edge))
                if edges
                else (lambda segment: False))

    def is_mouth(edge: QuadEdge,
                 cross_or_overlap_holes: Callable[[Segment], bool]
                 = to_edges_cross_or_overlap_detector(holes_edges),
                 segment_cls: Type[Segment] = context.segment_cls) -> bool:
        neighbour_end = edge.left_from_start.end
        return (neighbour_end not in boundary_points
                and not cross_or_overlap_holes(segment_cls(edge.start,
                                                           neighbour_end))
                and not cross_or_overlap_holes(segment_cls(edge.end,
                                                           neighbour_end)))

    def segments_cross_or_overlap(left: Segment, right: Segment) -> bool:
        relation = context.segments_relation(left, right)
        return (relation is not Relation.DISJOINT
                or relation is not Relation.TOUCH)

    edges_neighbours = {edge: to_edge_neighbours(edge)
                        for edge in boundary_edges}
    candidates = red_black.set_(*filter(is_mouth, boundary_edges),
                                key=_edge_key)
    boundary_vertices = [edge.start for edge in boundary_edges]
    compress_contour(boundary_vertices, context.angle_orientation)
    current_border_size = len(boundary_vertices)
    while current_border_size < border_size:
        try:
            edge = candidates.popmax()
        except ValueError:
            break
        if not is_mouth(edge):
            continue
        current_border_size += 1
        boundary_points.add(edge.left_from_start.end)
        triangulation.delete(edge)
        for neighbour in edges_neighbours.pop(edge):
            edges_neighbours[neighbour] = to_edge_neighbours(neighbour)
            candidates.add(neighbour)
    border_vertices = [edge.start for edge in to_boundary_edges(triangulation)]
    compress_contour(border_vertices, context.angle_orientation)
    return context.polygon_cls(contour_cls(border_vertices), holes)


def _reverse_vertices(vertices: List[Point[Scalar]]) -> List[Point[Scalar]]:
    return [vertices[0]] + vertices[:0:-1]


def _to_segment_angle(start: Point, end: Point) -> Real:
    return math.atan2(end.y - start.y, end.x - start.x)


def to_star_contour_vertices(points: Sequence[Point[Scalar]],
                             context: Context) -> Sequence[Point[Scalar]]:
    centroid = context.multipoint_centroid(context.multipoint_cls(points))
    contour_cls, region_centroid_constructor, orienteer = (
        context.contour_cls, context.region_centroid,
        context.angle_orientation)
    result, prev_size = points, len(points) + 1
    while 2 < len(result) < prev_size:
        prev_size = len(result)
        result = [deque(candidates,
                        maxlen=1)[0][1]
                  for _, candidates in groupby(sorted(
                    (_to_segment_angle(centroid, point), point)
                    for point in result),
                    key=itemgetter(0))]
        if len(result) > 2:
            centroid = region_centroid_constructor(contour_cls(result))
            index = 0
            while max(index, 2) < len(result):
                if not angle_contains_point(result[index], result[index - 1],
                                            result[(index + 1) % len(result)],
                                            centroid, orienteer):
                    del result[index]
                index += 1
            compress_contour(result, orienteer)
    return result


def to_convex_vertices_sequence(points: Sequence[Point[Scalar]],
                                random: Random,
                                context: Context) -> Sequence[Point[Scalar]]:
    """
    Based on Valtr algorithm by Sander Verdonschot.

    Time complexity:
        ``O(len(points) * log len(points))``
    Memory complexity:
        ``O(len(points))``
    Reference:
        http://cglab.ca/~sander/misc/ConvexGeneration/convex.html
    """
    xs, ys = [point.x for point in points], [point.y for point in points]
    xs, ys = sorted(xs), sorted(ys)
    min_x, *xs, max_x = xs
    min_y, *ys, max_y = ys

    def to_vectors_coordinates(coordinates: List[Scalar],
                               min_coordinate: Scalar,
                               max_coordinate: Scalar) -> List[Scalar]:
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

    def to_vector_angle(vector: Tuple[Scalar, Scalar]) -> Key:
        x, y = vector
        return atan2(y, x)

    vectors = sorted(zip(vectors_xs, vectors_ys),
                     key=to_vector_angle)
    point_x = point_y = 0
    min_polygon_x = min_polygon_y = 0
    coordinates_pairs = []
    for vector_x, vector_y in vectors:
        coordinates_pairs.append((point_x, point_y))
        point_x += vector_x
        point_y += vector_y
        min_polygon_x, min_polygon_y = (min(min_polygon_x, point_x),
                                        min(min_polygon_y, point_y))
    shift_x, shift_y = min_x - min_polygon_x, min_y - min_polygon_y
    point_cls = context.point_cls
    return context.points_convex_hull([point_cls(min(max(x + shift_x, min_x),
                                                     max_x),
                                                 min(max(y + shift_y, min_y),
                                                     max_y))
                                       for x, y in coordinates_pairs])


def compress_contour(vertices: MutableSequence[Point],
                     orienteer: Orienteer) -> None:
    index = -len(vertices) + 1
    while index < 0:
        while (max(2, -index) < len(vertices)
               and (orienteer(vertices[index + 1], vertices[index + 2],
                              vertices[index])
                    is Orientation.COLLINEAR)):
            del vertices[index + 1]
        index += 1
    while index < len(vertices):
        while (max(2, index) < len(vertices)
               and (orienteer(vertices[index - 1], vertices[index - 2],
                              vertices[index])
                    is Orientation.COLLINEAR)):
            del vertices[index - 1]
        index += 1


def to_max_convex_hull(points: Sequence[Point[Scalar]],
                       orienteer: Orienteer) -> Sequence[Point[Scalar]]:
    points = sorted(points)
    lower = _to_sub_hull(points, orienteer)
    upper = _to_sub_hull(reversed(points), orienteer)
    return lower[:-1] + upper[:-1]


def _to_sub_hull(points: Iterable[Point[Scalar]],
                 orienteer: Orienteer) -> List[Point[Scalar]]:
    result = []
    for point in points:
        while len(result) >= 2:
            if (orienteer(result[-2], result[-1], point)
                    is Orientation.CLOCKWISE):
                del result[-1]
            else:
                break
        result.append(point)
    return result


def contour_vertices_to_edges(vertices: Sequence[Point[Scalar]],
                              segment_cls: Type[Segment]
                              ) -> Sequence[Segment[Scalar]]:
    return [segment_cls(vertices[index - 1], vertices[index])
            for index in range(len(vertices))]


def to_vertices_sequence(points: Sequence[Point[Scalar]],
                         size: int,
                         context: Context) -> Sequence[Point[Scalar]]:
    """
    Based on chi-algorithm by M. Duckham et al.

    Time complexity:
        ``O(len(points) * log len(points))``
    Memory complexity:
        ``O(len(points))``
    Reference:
        http://www.geosensor.net/papers/duckham08.PR.pdf
    """
    triangulation = Triangulation.delaunay(points, context)
    boundary_edges = to_boundary_edges(triangulation)
    boundary_points = {edge.start for edge in boundary_edges}
    edges_increments = _to_edges_increments(boundary_edges)
    boundary_vertices = [edge.start for edge in boundary_edges]
    compress_contour(boundary_vertices, context.angle_orientation)
    edges_neighbours = {edge: to_edge_neighbours(edge)
                        for edge in boundary_edges}
    left_increment = size - len(boundary_vertices)
    while left_increment > 0:
        if any(edges_increments[1:]):
            max_increment = max(
                    increment
                    for increment, edges in enumerate(edges_increments[1:])
                    if edges
            )
            target_increment = min(max_increment, left_increment)
            candidates = edges_increments[target_increment + 1]
            for _ in range(len(candidates)):
                candidate = candidates.popmax()
                if not _is_mouth(candidate, boundary_points):
                    diagonal = candidate.left_from_end
                    if (diagonal.right_from_start.end not in boundary_points
                            and _is_convex_quadrilateral_diagonal(diagonal)):
                        diagonal.swap()
                        edges_neighbours[candidate] = to_edge_neighbours(
                                candidate
                        )
                    else:
                        del edges_neighbours[candidate]
                        continue
                actual_increment = _edge_to_increment(candidate)
                if actual_increment == target_increment:
                    break
                else:
                    edges_increments[actual_increment + 1].add(candidate)
            else:
                edges_increments = _to_edges_increments(edges_neighbours)
                continue
        else:
            candidates = edges_increments[0]
            for _ in range(len(candidates)):
                candidate = candidates.popmax()
                if _is_mouth(candidate, boundary_points):
                    diagonal = candidate.left_from_end
                    if (diagonal.right_from_start.end not in boundary_points
                            and _is_convex_quadrilateral_diagonal(diagonal)):
                        diagonal.swap()
                        edges_neighbours[candidate] = to_edge_neighbours(
                                candidate
                        )
                        break
                    else:
                        del edges_neighbours[candidate]
                        continue
            else:
                break
        assert _is_mouth(candidate, boundary_points)
        boundary_points.add(candidate.left_from_start.end)
        left_increment -= _edge_to_increment(candidate)
        triangulation.delete(candidate)
        for neighbour in edges_neighbours.pop(candidate):
            edges_neighbours[neighbour] = to_edge_neighbours(neighbour)
            edges_increments[_edge_to_increment(neighbour) + 1].add(
                    neighbour
            )
    return _triangulation_to_border_vertices(triangulation)


def _edge_to_increment(edge: QuadEdge) -> int:
    return (1
            - (edge.left_from_start.orientation_of(edge.right_from_start.end)
               is Orientation.COLLINEAR)
            - (edge.left_from_end.orientation_of(edge.right_from_end.end)
               is Orientation.COLLINEAR)
            + (edge.orientation_of(edge.right_from_start.end)
               is Orientation.COLLINEAR)
            + (edge.orientation_of(edge.right_from_end.end)
               is Orientation.COLLINEAR))


def _is_convex_quadrilateral_diagonal(edge: QuadEdge) -> bool:
    return (edge.right_from_start.orientation_of(edge.end)
            is Orientation.COUNTERCLOCKWISE
            is edge.right_from_end.opposite.orientation_of(
                    edge.left_from_start.end)
            is edge.left_from_end.orientation_of(edge.start)
            is edge.left_from_start.opposite.orientation_of(
                    edge.right_from_start.end))


def _is_mouth(edge: QuadEdge, boundary_points: Collection[Point]) -> bool:
    assert edge.start in boundary_points
    return edge.left_from_start.end not in boundary_points


def _to_edges_increments(edges: Iterable[QuadEdge]) -> Sequence[Set[QuadEdge]]:
    result = [red_black.set_(key=_edge_key) for _ in range(5)]
    for edge in edges:
        increment = _edge_to_increment(edge)
        result[increment + 1].add(edge)
    return result


def _triangulation_to_border_vertices(triangulation: Triangulation
                                      ) -> Sequence[Point[Scalar]]:
    result = [edge.start for edge in to_boundary_edges(triangulation)]
    compress_contour(result, triangulation.context.angle_orientation)
    return result
