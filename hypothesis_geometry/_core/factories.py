import math
from collections import deque
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    MutableSequence,
    Sequence,
)
from functools import partial
from itertools import groupby
from math import atan2
from operator import itemgetter
from random import Random
from typing import Any

from dendroid import red_black
from dendroid.hints import KeyedSet
from ground.context import Context
from ground.enums import Orientation, Relation
from ground.hints import Contour, Point, Polygon, Segment
from locus import segmental

from .constants import MIN_CONTOUR_SIZE
from .contracts import (
    angle_contains_point,
    has_horizontal_lowermost_segment,
    has_vertical_leftmost_segment,
)
from .hints import Chooser, Multicontour, Orienteer, ScalarT
from .subdivisional import QuadEdge, to_edge_neighbours
from .triangular import Triangulation, to_boundary_edges
from .utils import contours_do_not_cross_or_overlap


def to_multicontour(
    points: Sequence[Point[ScalarT]],
    sizes: Sequence[int],
    chooser: Chooser[Any],
    /,
    *,
    context: Context[ScalarT],
) -> Multicontour[ScalarT]:
    sorting_key_chooser = partial(
        chooser, [horizontal_point_key, vertical_point_key]
    )
    points = list(points)
    prior_sorting_key = predicate = None
    contour_cls, to_contour_segments = (
        context.contour_cls,
        context.contour_segments,
    )
    result = []
    for size in sizes:
        sorting_key = sorting_key_chooser()
        if sorting_key is not prior_sorting_key:
            prior_sorting_key, predicate = (
                sorting_key,
                has_vertical_leftmost_segment
                if sorting_key is horizontal_point_key
                else has_horizontal_lowermost_segment,
            )
            points.sort(key=sorting_key)
        contour_vertices = to_vertex_sequence(
            points[:size], size, context=context
        )
        if len(contour_vertices) >= MIN_CONTOUR_SIZE:
            contour = contour_cls(contour_vertices)
            result.append(contour)
            assert predicate is not None
            can_touch_next_contour = predicate(to_contour_segments(contour))
            points = points[size - can_touch_next_contour :]
    return result


def to_polygon(
    points: Sequence[Point[ScalarT]],
    border_size: int,
    hole_sizes: Sequence[int],
    chooser: Chooser[Any],
    /,
    *,
    context: Context[ScalarT],
) -> Polygon[ScalarT]:
    triangulation = Triangulation.delaunay(points, context)
    boundary_edges = to_boundary_edges(triangulation)
    boundary_points = {edge.start for edge in boundary_edges}
    sorting_key_chooser = partial(
        chooser, [horizontal_point_key, vertical_point_key]
    )
    inner_points = list(set(points) - boundary_points)
    prior_sorting_key = None
    predicate: Callable[[Sequence[Segment[ScalarT]]], bool] | None = None
    holes: list[Contour[ScalarT]] = []
    all_hole_edges: list[Segment[ScalarT]] = []
    contour_cls, to_contour_segments = (
        context.contour_cls,
        context.contour_segments,
    )
    for hole_size in hole_sizes:
        sorting_key = sorting_key_chooser()
        if sorting_key is not prior_sorting_key:
            prior_sorting_key, predicate = (
                sorting_key,
                (
                    has_vertical_leftmost_segment
                    if sorting_key is horizontal_point_key
                    else has_horizontal_lowermost_segment
                ),
            )
            inner_points.sort(key=sorting_key)
        hole_points = inner_points[:hole_size]
        hole_vertices = to_vertex_sequence(
            hole_points, hole_size, context=context
        )
        if len(hole_vertices) >= MIN_CONTOUR_SIZE:
            hole = contour_cls(_reverse_vertices(hole_vertices))
            assert contours_do_not_cross_or_overlap(
                [*holes, hole], context.contour_segments, context=context
            )
            holes.append(hole)
            hole_edges = to_contour_segments(hole)
            all_hole_edges.extend(hole_edges)
            boundary_points.update(hole_points)
            assert predicate is not None
            can_touch_next_hole = predicate(hole_edges)
            inner_points = inner_points[
                len(hole_points) - can_touch_next_hole :
            ]

    def to_cross_or_overlap_segment_detector(
        segments: Sequence[Segment[ScalarT]], /
    ) -> Callable[[Segment[ScalarT]], bool]:
        if len(segments) > 0:
            to_nearest_edge = segmental.Tree(
                segments, context=context
            ).nearest_segment

            def detector(
                segment: Segment[ScalarT],
                /,
                *,
                nearest_edge_to: Callable[
                    [Segment[ScalarT]], Segment[ScalarT]
                ] = to_nearest_edge,
            ) -> bool:
                return segments_cross_or_overlap(
                    nearest_edge_to(segment), segment
                )

            return detector
        return lambda _segment, /: False

    def is_mouth(
        edge: QuadEdge[ScalarT],
        /,
        *,
        cross_or_overlap_holes: Callable[[Segment[ScalarT]], bool],
        segment_cls: type[Segment[ScalarT]] = context.segment_cls,
    ) -> bool:
        neighbour_end = edge.left_from_start.end
        return (
            neighbour_end not in boundary_points
            and not cross_or_overlap_holes(
                segment_cls(edge.start, neighbour_end)
            )
            and not cross_or_overlap_holes(
                segment_cls(edge.end, neighbour_end)
            )
        )

    def segments_cross_or_overlap(
        left: Segment[ScalarT], right: Segment[ScalarT], /
    ) -> bool:
        relation = context.segments_relation(left, right)
        return not (
            relation is Relation.DISJOINT or relation is Relation.TOUCH
        )

    edges_cross_or_overlap_holes_detector = (
        to_cross_or_overlap_segment_detector(all_hole_edges)
    )
    edges_neighbours = {
        edge: to_edge_neighbours(edge) for edge in boundary_edges
    }
    edge_to_remove_candidates = red_black.set_(
        *filter(
            partial(
                is_mouth,
                cross_or_overlap_holes=edges_cross_or_overlap_holes_detector,
            ),
            boundary_edges,
        ),
        key=_edge_key,
    )
    boundary_vertices = [edge.start for edge in boundary_edges]
    compress_contour(boundary_vertices, orienteer=context.angle_orientation)
    current_border_size = len(boundary_vertices)
    while current_border_size < border_size:
        try:
            edge = edge_to_remove_candidates.popmax()
        except ValueError:
            break
        if not is_mouth(
            edge, cross_or_overlap_holes=edges_cross_or_overlap_holes_detector
        ):
            continue
        current_border_size += 1
        boundary_points.add(edge.left_from_start.end)
        triangulation.delete(edge)
        for neighbour in edges_neighbours.pop(edge):
            edges_neighbours[neighbour] = to_edge_neighbours(neighbour)
            edge_to_remove_candidates.add(neighbour)
    border_vertices = _triangulation_to_border_vertices(triangulation)
    assert len(border_vertices) >= MIN_CONTOUR_SIZE
    return context.polygon_cls(contour_cls(border_vertices), holes)


def _reverse_vertices(
    vertices: Sequence[Point[ScalarT]], /
) -> Sequence[Point[ScalarT]]:
    return [*vertices[:1], *vertices[:0:-1]]


def _to_segment_angle(start: Point[ScalarT], end: Point[ScalarT], /) -> float:
    return math.atan2(end.y - start.y, end.x - start.x)


def to_star_contour_vertices(
    points: Sequence[Point[ScalarT]], /, *, context: Context[ScalarT]
) -> Sequence[Point[ScalarT]]:
    centroid = context.multipoint_centroid(context.multipoint_cls(points))
    contour_cls, region_centroid_constructor, orienteer = (
        context.contour_cls,
        context.region_centroid,
        context.angle_orientation,
    )
    result, prev_size = points, len(points) + 1
    while 2 < len(result) < prev_size:
        prev_size = len(result)
        result = [
            deque(candidates, maxlen=1)[0][1]
            for _, candidates in groupby(
                sorted(
                    (_to_segment_angle(centroid, point), point)
                    for point in result
                ),
                key=itemgetter(0),
            )
        ]
        if len(result) > 2:
            centroid = region_centroid_constructor(contour_cls(result))
            index = 0
            while max(index, 2) < len(result):
                if not angle_contains_point(
                    result[index],
                    result[index - 1],
                    result[(index + 1) % len(result)],
                    centroid,
                    orienteer=orienteer,
                ):
                    del result[index]
                index += 1
            compress_contour(result, orienteer=orienteer)
    return result


def to_convex_vertex_sequence(
    points: Sequence[Point[ScalarT]],
    random: Random,
    /,
    *,
    context: Context[ScalarT],
) -> Sequence[Point[ScalarT]]:
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

    def to_vectors_coordinates(
        coordinates: list[ScalarT],
        min_coordinate: ScalarT,
        max_coordinate: ScalarT,
        /,
    ) -> list[ScalarT]:
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

    def to_vector_angle(vector: tuple[ScalarT, ScalarT], /) -> float:
        x, y = vector
        return atan2(y, x)

    vectors = sorted(
        zip(vectors_xs, vectors_ys, strict=True), key=to_vector_angle
    )
    point_x = point_y = context.zero
    min_polygon_x = min_polygon_y = context.zero
    coordinates_pairs = []
    for vector_x, vector_y in vectors:
        coordinates_pairs.append((point_x, point_y))
        point_x += vector_x
        point_y += vector_y
        min_polygon_x, min_polygon_y = (
            min(min_polygon_x, point_x),
            min(min_polygon_y, point_y),
        )
    shift_x, shift_y = min_x - min_polygon_x, min_y - min_polygon_y
    point_cls = context.point_cls
    return context.points_convex_hull(
        [
            point_cls(
                min(max(x + shift_x, min_x), max_x),
                min(max(y + shift_y, min_y), max_y),
            )
            for x, y in coordinates_pairs
        ]
    )


def compress_contour(
    vertices: MutableSequence[Point[ScalarT]],
    /,
    *,
    orienteer: Orienteer[ScalarT],
) -> None:
    index = -len(vertices) + 1
    while index < 0:
        while max(2, -index) < len(vertices) and (
            orienteer(
                vertices[index + 1], vertices[index + 2], vertices[index]
            )
            is Orientation.COLLINEAR
        ):
            del vertices[index + 1]
        index += 1
    while index < len(vertices):
        while max(2, index) < len(vertices) and (
            orienteer(
                vertices[index - 1], vertices[index - 2], vertices[index]
            )
            is Orientation.COLLINEAR
        ):
            del vertices[index - 1]
        index += 1


def to_max_convex_hull(
    points: Sequence[Point[ScalarT]], /, *, orienteer: Orienteer[ScalarT]
) -> Sequence[Point[ScalarT]]:
    points = sorted(points)
    lower = _to_sub_hull(points, orienteer=orienteer)
    upper = _to_sub_hull(reversed(points), orienteer=orienteer)
    return lower[:-1] + upper[:-1]


def _to_sub_hull(
    points: Iterable[Point[ScalarT]], /, *, orienteer: Orienteer[ScalarT]
) -> list[Point[ScalarT]]:
    result: list[Point[ScalarT]] = []
    for point in points:
        while len(result) >= 2:
            if (
                orienteer(result[-2], result[-1], point)
                is Orientation.CLOCKWISE
            ):
                del result[-1]
            else:
                break
        result.append(point)
    return result


def contour_vertices_to_edges(
    vertices: Sequence[Point[ScalarT]],
    /,
    *,
    segment_cls: type[Segment[ScalarT]],
) -> Sequence[Segment[ScalarT]]:
    return [
        segment_cls(vertices[index - 1], vertices[index])
        for index in range(len(vertices))
    ]


def to_vertex_sequence(
    points: Sequence[Point[ScalarT]],
    size: int,
    /,
    *,
    context: Context[ScalarT],
) -> Sequence[Point[ScalarT]]:
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
    return _to_vertex_sequence(
        triangulation, boundary_edges, size, context=context
    )


def _to_vertex_sequence(
    triangulation: Triangulation[ScalarT],
    boundary_edges: Sequence[QuadEdge[ScalarT]],
    size: int,
    /,
    *,
    context: Context[ScalarT],
) -> Sequence[Point[ScalarT]]:
    boundary_points = {edge.start for edge in boundary_edges}
    boundary_vertices = [edge.start for edge in boundary_edges]
    compress_contour(boundary_vertices, orienteer=context.angle_orientation)
    if len(boundary_vertices) < MIN_CONTOUR_SIZE:
        return boundary_vertices
    mouths_increments = _to_mouths_increments(boundary_edges)
    mouths_candidates = set(boundary_edges)
    left_increment = size - len(boundary_vertices)
    while left_increment > 0:
        target_increment = max(
            [
                increment
                for increment, edges in enumerate(
                    mouths_increments[MAX_MOUTH_DECREMENT:]
                )
                if edges and increment <= left_increment
            ],
            default=None,
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
                    if (
                        diagonal.right_from_start.end not in boundary_points
                        and _is_convex_quadrilateral_diagonal(diagonal)
                    ):
                        diagonal.flip()
                        actual_increment = _mouth_to_increment(candidate)
                        break
                    mouths_candidates.remove(candidate)
                    continue
            else:
                break
        else:
            candidates = mouths_increments[
                target_increment + MAX_MOUTH_DECREMENT
            ]
            for _ in range(len(candidates)):
                candidate = candidates.popmax()
                if not _is_mouth(candidate, boundary_points):
                    diagonal = candidate.left_from_end
                    if (
                        diagonal.right_from_start.end not in boundary_points
                        and _is_convex_quadrilateral_diagonal(diagonal)
                    ):
                        diagonal.flip()
                    else:
                        mouths_candidates.remove(candidate)
                        continue
                actual_increment = _mouth_to_increment(candidate)
                if actual_increment == target_increment:
                    break
                (
                    mouths_increments[
                        actual_increment + MAX_MOUTH_DECREMENT
                    ].add(candidate)
                )
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
    ears_candidates = (
        set(to_boundary_edges(triangulation)) if left_increment > 0 else set()
    )
    ears_increments = _to_ears_increments(ears_candidates)
    while left_increment > 0:
        target_increment = max(
            [
                increment
                for increment, edges in enumerate(
                    ears_increments[MAX_EAR_DECREMENT:]
                )
                if edges and increment <= left_increment
            ],
            default=None,
        )
        if target_increment is None:
            break
        candidates = ears_increments[target_increment + MAX_EAR_DECREMENT]
        for _ in range(len(candidates)):
            candidate = candidates.popmax()
            if not _is_ear(candidate):
                ears_candidates.remove(candidate)
                continue
            actual_increment = _ear_to_increment(candidate)
            if actual_increment == target_increment:
                break
            (
                ears_increments[actual_increment + MAX_EAR_DECREMENT].add(
                    candidate
                )
            )
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


def _edge_key(
    edge: QuadEdge[ScalarT], /
) -> tuple[int, Point[ScalarT], Point[ScalarT]]:
    weight = 0
    adjacent_edges = []
    cursor = edge
    while True:
        diagonal = _is_convex_quadrilateral_diagonal(cursor.left_from_end)
        assert (
            not diagonal or cursor.left_from_end is not cursor.right_from_end
        )
        weight += diagonal
        cursor = cursor.left_from_start
        adjacent_edges.append(cursor)
        if cursor is edge:
            break
    return weight, edge.start, edge.end


def to_edges(edge: QuadEdge[ScalarT], /) -> Iterable[QuadEdge[ScalarT]]:
    visited_edges: set[QuadEdge[ScalarT]] = set()
    is_visited, visit_multiple = (
        visited_edges.__contains__,
        visited_edges.update,
    )
    queue = [edge]
    while queue:
        edge = queue.pop()
        if is_visited(edge):
            continue
        yield edge
        visit_multiple((edge, edge.opposite))
        queue.extend(
            (
                edge.left_from_start,
                edge.left_from_end,
                edge.right_from_start,
                edge.right_from_end,
            )
        )


def _mouth_to_increment(edge: QuadEdge[ScalarT], /) -> int:
    return (
        1
        - (
            edge.left_from_start.orientation_of(edge.right_from_start.end)
            is Orientation.COLLINEAR
        )
        - (
            edge.left_from_end.orientation_of(edge.right_from_end.end)
            is Orientation.COLLINEAR
        )
        + (
            edge.orientation_of(edge.right_from_start.end)
            is Orientation.COLLINEAR
        )
        + (
            edge.orientation_of(edge.right_from_end.end)
            is Orientation.COLLINEAR
        )
    )


def _ear_to_increment(edge: QuadEdge[ScalarT], /) -> int:
    return (
        (
            edge.right_from_end.orientation_of(
                edge.right_from_end.right_from_end.end
            )
            is Orientation.COLLINEAR
        )
        + (
            edge.orientation_of(edge.right_from_start.end)
            is Orientation.COLLINEAR
        )
        - (
            edge.right_from_start.orientation_of(edge.right_from_end.end)
            is Orientation.COLLINEAR
        )
        - (
            edge.right_from_end.right_from_end.orientation_of(edge.start)
            is Orientation.COLLINEAR
        )
        - 1
    )


def _is_convex_quadrilateral_diagonal(edge: QuadEdge[ScalarT], /) -> bool:
    return (
        edge.right_from_start.orientation_of(edge.end)
        is Orientation.COUNTERCLOCKWISE
        is edge.right_from_end.opposite.orientation_of(
            edge.left_from_start.end
        )
        is edge.left_from_end.orientation_of(edge.start)
        is edge.left_from_start.opposite.orientation_of(
            edge.right_from_start.end
        )
    )


def _is_ear(edge: QuadEdge[ScalarT], /) -> bool:
    return (
        edge.orientation_of(edge.right_from_end.end)
        is Orientation.COUNTERCLOCKWISE
    ) and _is_convex_quadrilateral_diagonal(
        edge.left_from_start
        if edge.left_from_end is edge.right_from_end
        else edge.left_from_end
    )


def _is_mouth(
    edge: QuadEdge[ScalarT], boundary_points: Collection[Point[ScalarT]], /
) -> bool:
    assert edge.start in boundary_points
    return edge.left_from_start.end not in boundary_points


MAX_EAR_DECREMENT = 3
MAX_EAR_INCREMENT = 1


def _to_ears_increments(
    edges: Iterable[QuadEdge[ScalarT]], /
) -> Sequence[KeyedSet[Any, QuadEdge[ScalarT]]]:
    result: list[KeyedSet[Any, QuadEdge[ScalarT]]] = [
        red_black.set_(key=_edge_key)
        for _ in range(-MAX_EAR_DECREMENT, MAX_EAR_INCREMENT + 1)
    ]
    for edge in edges:
        increment = _ear_to_increment(edge)
        result[increment + MAX_EAR_DECREMENT].add(edge)
    return result


MAX_MOUTH_DECREMENT = 1
MAX_MOUTH_INCREMENT = 3


def _to_mouths_increments(
    edges: Iterable[QuadEdge[ScalarT]], /
) -> Sequence[KeyedSet[Any, QuadEdge[ScalarT]]]:
    result: list[KeyedSet[Any, QuadEdge[ScalarT]]] = [
        red_black.set_(key=_edge_key)
        for _ in range(-MAX_MOUTH_DECREMENT, MAX_MOUTH_INCREMENT + 1)
    ]
    for edge in edges:
        increment = _mouth_to_increment(edge)
        result[increment + MAX_MOUTH_DECREMENT].add(edge)
    return result


def _triangulation_to_border_vertices(
    triangulation: Triangulation[ScalarT], /
) -> Sequence[Point[ScalarT]]:
    result = [edge.start for edge in to_boundary_edges(triangulation)]
    compress_contour(result, orienteer=triangulation.context.angle_orientation)
    return result


def horizontal_point_key(point: Point[ScalarT], /) -> tuple[ScalarT, ScalarT]:
    return point.x, point.y


def vertical_point_key(point: Point[ScalarT], /) -> tuple[ScalarT, ScalarT]:
    return point.y, point.x
