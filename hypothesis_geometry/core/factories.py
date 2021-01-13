import math
from collections import deque
from functools import partial
from itertools import (cycle,
                       groupby)
from math import atan2
from operator import (attrgetter,
                      itemgetter)
from random import Random
from typing import (Callable,
                    Iterable,
                    List,
                    MutableSequence,
                    Optional,
                    Sequence,
                    Tuple,
                    Type)

from dendroid import red_black
from dendroid.hints import Key
from ground.base import (Context,
                         Orientation,
                         Relation)
from ground.hints import (Contour,
                          Coordinate,
                          Point,
                          Polygon,
                          Segment)
from locus import segmental

from hypothesis_geometry.hints import Multicontour
from .contracts import (has_horizontal_lowermost_segment,
                        has_vertical_leftmost_segment,
                        to_angle_containment_detector)
from .hints import (CentroidConstructor,
                    Chooser,
                    ContourCompressor,
                    ContourEdgesConstructor,
                    Orienteer,
                    PointsSequenceOperator,
                    PolygonEdgesConstructor,
                    QuadEdge,
                    QuaternaryPointFunction,
                    Triangulation)
from .subdivisional import to_edge_neighbours
from .triangular import (to_boundary_edges,
                         to_triangulation_cls)


def to_contour_compressor(context: Context) -> ContourCompressor:
    return partial(_compress_contour, context.angle_orientation)


def to_contour_edges_constructor(context: Context) -> ContourEdgesConstructor:
    return partial(_contour_to_edges, context.segment_cls)


def to_convex_hull_size_constrictor(context: Context,
                                    *,
                                    max_size: Optional[int]
                                    ) -> PointsSequenceOperator:
    return partial(_constrict_convex_hull_size, context.points_convex_hull,
                   to_max_convex_hull_constructor(context),
                   context.angle_orientation,
                   max_size=max_size)


def to_convex_vertices_sequence_factory(context: Context
                                        ) -> Callable[[Sequence[Point],
                                                       Random],
                                                      Sequence[Point]]:
    return partial(_to_convex_vertices_sequence, context.points_convex_hull,
                   context.point_cls)


def to_max_convex_hull_constructor(context: Context) -> PointsSequenceOperator:
    return partial(_to_max_convex_hull, context.angle_orientation)


def to_multicontour_factory(context: Context
                            ) -> Callable[[Sequence[Point], Sequence[int],
                                           Chooser], Multicontour]:
    return partial(_to_multicontour, to_vertices_sequence_factory(context),
                   context.contour_cls, to_contour_edges_constructor(context))


def to_polygon_border_edges_constructor(context: Context
                                        ) -> PolygonEdgesConstructor:
    return partial(_polygon_to_border_edges,
                   to_contour_edges_constructor(context))


def to_polygon_factory(context: Context
                       ) -> Callable[[Sequence[Point], int, List[int],
                                      Chooser], Polygon]:
    return partial(_to_polygon, context.contour_cls,
                   to_contour_compressor(context),
                   to_contour_edges_constructor(context), context.polygon_cls,
                   context.segment_cls, context.segments_relation,
                   to_triangulation_cls(context),
                   to_vertices_sequence_factory(context))


def to_star_contour_vertices_factory(context: Context
                                     ) -> PointsSequenceOperator:
    return partial(_to_star_contour_vertices,
                   to_angle_containment_detector(context),
                   to_contour_compressor(context),
                   context.contour_centroid, context.multipoint_centroid)


def to_vertices_sequence_factory(context: Context
                                 ) -> Callable[[Sequence[Point], int],
                                               Sequence[Point]]:
    return partial(_to_vertices_sequence, to_contour_compressor(context),
                   to_triangulation_cls(context))


def _constrict_convex_hull_size(convex_hull_constructor
                                : PointsSequenceOperator,
                                max_convex_hull_constructor
                                : PointsSequenceOperator,
                                orienteer: Orienteer,
                                points: Sequence[Point],
                                *,
                                max_size: Optional[int]) -> Sequence[Point]:
    if max_size is None:
        return points
    convex_hull = convex_hull_constructor(points)
    if len(convex_hull) <= max_size:
        return points
    sorted_convex_hull = sorted(
            convex_hull,
            key=partial(_to_squared_points_distance, convex_hull[0]))
    new_border_points = []
    for index in range(max_size):
        quotient, remainder = divmod(index, 2)
        new_border_points.append(sorted_convex_hull[-quotient - 1]
                                 if remainder
                                 else sorted_convex_hull[quotient])
    new_border = list(max_convex_hull_constructor(new_border_points))
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
    return _to_squared_edge_length(edge), edge.start, edge.end


def _to_multicontour(contour_vertices_factory
                     : Callable[[Sequence[Point], int], Sequence[Point]],
                     contour_cls: Type[Contour],
                     contour_edges_constructor: ContourEdgesConstructor,
                     points: Sequence[Point],
                     sizes: Sequence[int],
                     chooser: Chooser) -> Multicontour:
    sorting_key_chooser = partial(chooser, [None, attrgetter('y', 'x'),
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
    result = []
    for size in sizes:
        contour_vertices = contour_vertices_factory(points[:size], size)
        result.append(contour_cls(contour_vertices))
        can_touch_next_contour = current_predicate(
                contour_edges_constructor(contour_vertices))
        points = points[size - can_touch_next_contour:]
        new_sorting_key = sorting_key_chooser()
        if new_sorting_key is not current_sorting_key:
            (points, current_sorting_key,
             current_predicate) = (sorted(points,
                                          key=new_sorting_key),
                                   new_sorting_key, next(predicates))
    return result


def _to_polygon(contour_cls: Type[Contour],
                contour_compressor: ContourCompressor,
                contour_edges_constructor: ContourEdgesConstructor,
                polygon_cls: Type[Polygon],
                segment_cls: Type[Segment],
                segments_relater: QuaternaryPointFunction[Relation],
                triangulation_cls: Type[Triangulation],
                vertices_sequence_factory: Callable[[Sequence[Point], int],
                                                    Sequence[Point]],
                points: Sequence[Point],
                border_size: int,
                holes_sizes: List[int],
                chooser: Chooser) -> Polygon:
    triangulation = triangulation_cls.delaunay(points)
    boundary_edges = to_boundary_edges(triangulation)
    boundary_points = {edge.start for edge in boundary_edges}
    sorting_key_chooser = partial(chooser, [None, attrgetter('y', 'x'),
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
    holes = []
    holes_segments = []
    for hole_size in holes_sizes:
        hole_points = inner_points[:hole_size]
        hole_vertices = vertices_sequence_factory(hole_points, hole_size)[::-1]
        holes.append(contour_cls(hole_vertices))
        hole_segments = contour_edges_constructor(hole_vertices)
        holes_segments.extend(hole_segments)
        boundary_points.update(hole_points)
        can_touch_next_hole = current_predicate(hole_segments)
        inner_points = inner_points[hole_size - can_touch_next_hole:]
        next_sorting_key = sorting_key_chooser()
        if next_sorting_key is not current_sorting_key:
            (current_sorting_key, inner_points,
             current_predicate) = (next_sorting_key,
                                   sorted(inner_points,
                                          key=next_sorting_key),
                                   next(predicates))

    def to_segment_cross_or_overlap_detector(segments: Sequence[Segment]
                                             ) -> Callable[[Segment], bool]:
        return ((lambda segment, to_nearest_segment=(segmental.Tree(segments)
                                                     .nearest_segment)
                 : segments_cross_or_overlap(to_nearest_segment(segment),
                                             segment))
                if segments
                else (lambda segment: False))

    def is_mouth(edge: QuadEdge,
                 cross_or_overlap_holes: Callable[[Segment], bool]
                 = to_segment_cross_or_overlap_detector(holes_segments)
                 ) -> bool:
        neighbour_end = edge.left_from_start.end
        return (neighbour_end not in boundary_points
                and not cross_or_overlap_holes(segment_cls(edge.start,
                                                           neighbour_end))
                and not cross_or_overlap_holes(segment_cls(edge.end,
                                                           neighbour_end)))

    def segments_cross_or_overlap(left: Segment, right: Segment) -> bool:
        relation = segments_relater(left.start, left.end, right.start,
                                    right.end)
        return (relation is not Relation.DISJOINT
                or relation is not Relation.TOUCH)

    edges_neighbours = {edge: to_edge_neighbours(edge)
                        for edge in boundary_edges}
    candidates = red_black.set_(*filter(is_mouth, boundary_edges),
                                key=_edge_key)
    boundary_vertices = [edge.start for edge in boundary_edges]
    contour_compressor(boundary_vertices)
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
    contour_compressor(border_vertices)
    return polygon_cls(contour_cls(border_vertices), holes)


def _to_segment_angle(start: Point, end: Point) -> Coordinate:
    return math.atan2(end.y - start.y, end.x - start.x)


def _to_star_contour_vertices(angle_containment_detector
                              : QuaternaryPointFunction[bool],
                              contour_compressor: ContourCompressor,
                              contour_centroid_constructor
                              : CentroidConstructor,
                              multipoint_centroid_constructor
                              : CentroidConstructor,
                              points: Sequence[Point]) -> Sequence[Point]:
    centroid = multipoint_centroid_constructor(points)
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
            centroid = contour_centroid_constructor(result)
            index = 0
            while max(index, 2) < len(result):
                if not angle_containment_detector(
                        result[index], result[index - 1],
                        result[(index + 1) % len(result)], centroid):
                    del result[index]
                index += 1
            contour_compressor(result)
    return result


def _to_squared_edge_length(edge: QuadEdge) -> Coordinate:
    return _to_squared_points_distance(edge.start, edge.end)


def _to_squared_points_distance(left: Point, right: Point) -> Coordinate:
    return (left.x - right.x) ** 2 + (left.y - right.y) ** 2


def _to_convex_vertices_sequence(convex_hull_constructor
                                 : PointsSequenceOperator,
                                 point_cls: Type[Point],
                                 points: Sequence[Point],
                                 random: Random) -> Sequence[Point]:
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

    def to_vector_angle(vector: Tuple[Coordinate, Coordinate]) -> Key:
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
    return convex_hull_constructor([point_cls(min(max(x + shift_x, min_x),
                                                  max_x),
                                              min(max(y + shift_y, min_y),
                                                  max_y))
                                    for x, y in coordinates_pairs])


def _compress_contour(orienteer: Orienteer,
                      vertices: MutableSequence[Point]) -> None:
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


def _to_max_convex_hull(orienteer: Orienteer,
                        points: Sequence[Point]) -> Sequence[Point]:
    points = sorted(points)
    lower = _to_sub_hull(orienteer, points)
    upper = _to_sub_hull(orienteer, reversed(points))
    return lower[:-1] + upper[:-1]


def _to_sub_hull(orienteer: Orienteer, points: Iterable[Point]) -> List[Point]:
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


def _contour_to_edges(segments_cls: Type[Segment],
                      vertices: Sequence[Point]) -> Sequence[Segment]:
    return [segments_cls(vertices[index - 1], vertices[index])
            for index in range(len(vertices))]


def _polygon_to_border_edges(contour_edges_constructor
                             : ContourEdgesConstructor,
                             polygon: Polygon) -> Sequence[Segment]:
    return contour_edges_constructor(polygon.border.vertices)


def _to_vertices_sequence(contour_compressor: ContourCompressor,
                          triangulation_cls: Type[Triangulation],
                          points: Sequence[Point],
                          size: int) -> Sequence[Point]:
    """
    Based on chi-algorithm by M. Duckham et al.

    Time complexity:
        ``O(len(points) * log len(points))``
    Memory complexity:
        ``O(len(points))``
    Reference:
        http://www.geosensor.net/papers/duckham08.PR.pdf
    """
    triangulation = triangulation_cls.delaunay(points)
    boundary_edges = to_boundary_edges(triangulation)
    boundary_points = {edge.start for edge in boundary_edges}

    def is_mouth(edge: QuadEdge) -> bool:
        return edge.left_from_start.end not in boundary_points

    edges_neighbours = {edge: to_edge_neighbours(edge)
                        for edge in boundary_edges}
    candidates = red_black.set_(*filter(is_mouth, boundary_edges),
                                key=_edge_key)
    boundary_vertices = [edge.start for edge in boundary_edges]
    contour_compressor(boundary_vertices)
    current_size = len(boundary_vertices)
    while current_size < size:
        try:
            edge = candidates.popmax()
        except ValueError:
            break
        if not is_mouth(edge):
            continue
        size += 1
        boundary_points.add(edge.left_from_start.end)
        triangulation.delete(edge)
        for neighbour in edges_neighbours.pop(edge):
            edges_neighbours[neighbour] = to_edge_neighbours(neighbour)
            candidates.add(neighbour)
    result = [edge.start for edge in to_boundary_edges(triangulation)]
    contour_compressor(result)
    return result
