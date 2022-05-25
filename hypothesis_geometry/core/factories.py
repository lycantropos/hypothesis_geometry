import math
from collections import deque
from functools import partial
from itertools import groupby
from math import atan2
from numbers import Real
from operator import itemgetter
from random import Random
from typing import (Callable,
                    Collection,
                    Iterable,
                    List,
                    MutableSequence,
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

from .constants import MIN_CONTOUR_SIZE
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


def to_multicontour(points: Sequence[Point[Scalar]],
                    sizes: Sequence[int],
                    chooser: Chooser,
                    context: Context) -> Multicontour[Scalar]:
    sorting_key_chooser = partial(chooser, [horizontal_point_key,
                                            vertical_point_key])
    points = list(points)
    prior_sorting_key = predicate = None
    contour_cls, to_contour_segments = (context.contour_cls,
                                        context.contour_segments)
    result = []
    for size in sizes:
        sorting_key = sorting_key_chooser()
        if sorting_key is not prior_sorting_key:
            prior_sorting_key, predicate = (
                sorting_key,
                has_vertical_leftmost_segment
                if sorting_key is horizontal_point_key
                else has_horizontal_lowermost_segment
            )
            points.sort(key=sorting_key)
        contour_vertices = to_vertices_sequence(points[:size], size, context)
        if len(contour_vertices) >= MIN_CONTOUR_SIZE:
            contour = contour_cls(contour_vertices)
            result.append(contour)
            can_touch_next_contour = predicate(to_contour_segments(contour))
            points = points[size - can_touch_next_contour:]
    return result


def to_polygon(points: Sequence[Point[Scalar]],
               border_size: int,
               holes_sizes: List[int],
               chooser: Chooser,
               context: Context) -> Polygon[Scalar]:
    triangulation = Triangulation.delaunay(points, context)
    boundary_edges = to_boundary_edges(triangulation)
    boundary_points = {edge.start for edge in boundary_edges}
    sorting_key_chooser = partial(chooser, [horizontal_point_key,
                                            vertical_point_key])
    inner_points = list(set(points) - boundary_points)
    prior_sorting_key = predicate = None
    holes, holes_edges = [], []
    contour_cls, to_contour_segments = (context.contour_cls,
                                        context.contour_segments)
    for hole_size in holes_sizes:
        sorting_key = sorting_key_chooser()
        if sorting_key is not prior_sorting_key:
            prior_sorting_key, predicate = (
                sorting_key,
                has_vertical_leftmost_segment
                if sorting_key is horizontal_point_key
                else has_horizontal_lowermost_segment
            )
            inner_points.sort(key=sorting_key)
        hole_points = inner_points[:hole_size]
        hole_vertices = to_vertices_sequence(hole_points, hole_size, context)
        if len(hole_vertices) >= MIN_CONTOUR_SIZE:
            hole = contour_cls(_reverse_vertices(hole_vertices))
            holes.append(hole)
            hole_edges = to_contour_segments(hole)
            holes_edges.extend(hole_edges)
            boundary_points.update(hole_points)
            can_touch_next_hole = predicate(hole_edges)
            inner_points = inner_points[len(hole_points)
                                        - can_touch_next_hole:]

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
    border_vertices = _triangulation_to_border_vertices(triangulation)
    assert len(border_vertices) >= MIN_CONTOUR_SIZE
    return context.polygon_cls(contour_cls(border_vertices), holes)


def _reverse_vertices(vertices: Sequence[Point[Scalar]]
                      ) -> Sequence[Point[Scalar]]:
    return vertices[:1] + vertices[:0:-1]


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
    boundary_vertices = [edge.start for edge in boundary_edges]
    compress_contour(boundary_vertices, context.angle_orientation)
    if len(boundary_vertices) < MIN_CONTOUR_SIZE:
        return boundary_vertices
    mouths_increments = _to_mouths_increments(boundary_edges)
    mouths_candidates = set(boundary_edges)
    left_increment = size - len(boundary_vertices)
    while left_increment > 0:
        target_increment = max(
                [increment
                 for increment, edges
                 in enumerate(mouths_increments[MAX_MOUTH_DECREMENT:])
                 if edges and increment <= left_increment],
                default=None
        )
        if target_increment is None:
            target_increment = 1
            candidates = mouths_increments[0]
            for _ in range(len(candidates)):
                candidate = candidates.popmax()
                actual_increment = _mouth_to_increment(candidate)
                assert actual_increment == -1
                if _is_mouth(candidate, boundary_points):
                    diagonal = candidate.left_from_end
                    if (diagonal.right_from_start.end not in boundary_points
                            and _is_convex_quadrilateral_diagonal(diagonal)):
                        diagonal.flip()
                        actual_increment = _mouth_to_increment(candidate)
                        break
                    else:
                        mouths_candidates.remove(candidate)
                        continue
            else:
                break
        else:
            candidates = mouths_increments[target_increment
                                           + MAX_MOUTH_DECREMENT]
            for _ in range(len(candidates)):
                candidate = candidates.popmax()
                if not _is_mouth(candidate, boundary_points):
                    diagonal = candidate.left_from_end
                    if (diagonal.right_from_start.end not in boundary_points
                            and _is_convex_quadrilateral_diagonal(diagonal)):
                        diagonal.flip()
                    else:
                        mouths_candidates.remove(candidate)
                        continue
                actual_increment = _mouth_to_increment(candidate)
                if actual_increment == target_increment:
                    break
                else:
                    (mouths_increments[actual_increment + MAX_MOUTH_DECREMENT]
                     .add(candidate))
            else:
                mouths_increments = _to_mouths_increments(mouths_candidates)
                continue
        assert actual_increment == target_increment
        assert _is_mouth(candidate, boundary_points)
        boundary_points.add(candidate.left_from_start.end)
        left_increment -= actual_increment
        neighbours = to_edge_neighbours(candidate)
        mouths_candidates.remove(candidate)
        triangulation.delete(candidate)
        for neighbour in neighbours:
            mouths_candidates.add(neighbour)
            increment = _mouth_to_increment(neighbour)
            mouths_increments[increment + MAX_MOUTH_DECREMENT].add(neighbour)
    ears_candidates = (set(to_boundary_edges(triangulation))
                       if left_increment > 0
                       else set())
    ears_increments = _to_ears_increments(ears_candidates)
    while left_increment > 0:
        target_increment = max(
                [increment
                 for increment, edges
                 in enumerate(ears_increments[MAX_EAR_DECREMENT:])
                 if edges and increment <= left_increment],
                default=None
        )
        if target_increment is None:
            break
        else:
            candidates = ears_increments[target_increment + MAX_EAR_DECREMENT]
            for _ in range(len(candidates)):
                candidate = candidates.popmax()
                if not _is_ear(candidate):
                    ears_candidates.remove(candidate)
                    continue
                actual_increment = _ear_to_increment(candidate)
                if actual_increment == target_increment:
                    break
                else:
                    (ears_increments[actual_increment + MAX_EAR_DECREMENT]
                     .add(candidate))
            else:
                ears_increments = _to_ears_increments(ears_candidates)
                continue
        while candidate.left_from_end is not candidate.right_from_end:
            candidate.left_from_end.flip()
        assert actual_increment == target_increment
        left_increment -= actual_increment
        ears_candidates.remove(candidate)
        ear_base = candidate.left_from_start
        triangulation.delete(candidate)
        candidate.right_from_end.flip()
        ears_candidates.add(ear_base)
    return _triangulation_to_border_vertices(triangulation)


def _edge_key(edge: QuadEdge) -> Key:
    weight = 0
    cursor = edge
    while True:
        weight += _is_convex_quadrilateral_diagonal(cursor.left_from_end)
        cursor = cursor.left_from_start
        if cursor is edge:
            break
    return weight, edge.start, edge.end


def _mouth_to_increment(edge: QuadEdge) -> int:
    return (1
            - (edge.left_from_start.orientation_of(edge.right_from_start.end)
               is Orientation.COLLINEAR)
            - (edge.left_from_end.orientation_of(edge.right_from_end.end)
               is Orientation.COLLINEAR)
            + (edge.orientation_of(edge.right_from_start.end)
               is Orientation.COLLINEAR)
            + (edge.orientation_of(edge.right_from_end.end)
               is Orientation.COLLINEAR))


def _ear_to_increment(edge: QuadEdge) -> int:
    return ((edge.right_from_end.orientation_of(edge.right_from_end
                                                .right_from_end.end)
             is Orientation.COLLINEAR)
            + (edge.orientation_of(edge.right_from_start.end)
               is Orientation.COLLINEAR)
            - (edge.right_from_start.orientation_of(edge.right_from_end.end)
               is Orientation.COLLINEAR)
            - (edge.right_from_end.right_from_end.orientation_of(edge.start)
               is Orientation.COLLINEAR)
            - 1)


def _is_convex_quadrilateral_diagonal(edge: QuadEdge) -> bool:
    return (edge.right_from_start.orientation_of(edge.end)
            is Orientation.COUNTERCLOCKWISE
            is edge.right_from_end.opposite.orientation_of(
                    edge.left_from_start.end)
            is edge.left_from_end.orientation_of(edge.start)
            is edge.left_from_start.opposite.orientation_of(
                    edge.right_from_start.end))


def _is_ear(edge: QuadEdge) -> bool:
    return ((edge.orientation_of(edge.right_from_end.end)
             is Orientation.COUNTERCLOCKWISE)
            and _is_convex_quadrilateral_diagonal(
                    edge.left_from_start
                    if edge.left_from_end is edge.right_from_end
                    else edge.left_from_end
            ))


def _is_mouth(edge: QuadEdge, boundary_points: Collection[Point]) -> bool:
    assert edge.start in boundary_points
    return edge.left_from_start.end not in boundary_points


MAX_EAR_DECREMENT = 3
MAX_EAR_INCREMENT = 1


def _to_ears_increments(edges: Iterable[QuadEdge]) -> Sequence[Set[QuadEdge]]:
    result = [red_black.set_(key=_edge_key)
              for _ in range(-MAX_EAR_DECREMENT, MAX_EAR_INCREMENT + 1)]
    for edge in edges:
        increment = _ear_to_increment(edge)
        result[increment + MAX_EAR_DECREMENT].add(edge)
    return result


MAX_MOUTH_DECREMENT = 1
MAX_MOUTH_INCREMENT = 3


def _to_mouths_increments(edges: Iterable[QuadEdge]
                          ) -> Sequence[Set[QuadEdge]]:
    result = [red_black.set_(key=_edge_key)
              for _ in range(-MAX_MOUTH_DECREMENT, MAX_MOUTH_INCREMENT + 1)]
    for edge in edges:
        increment = _mouth_to_increment(edge)
        result[increment + MAX_MOUTH_DECREMENT].add(edge)
    return result


def _triangulation_to_border_vertices(triangulation: Triangulation
                                      ) -> Sequence[Point[Scalar]]:
    result = [edge.start for edge in to_boundary_edges(triangulation)]
    compress_contour(result, triangulation.context.angle_orientation)
    return result


def horizontal_point_key(point: Point[Scalar]) -> Tuple[Scalar, Scalar]:
    return point.x, point.y


def vertical_point_key(point: Point[Scalar]) -> Tuple[Scalar, Scalar]:
    return point.y, point.x
