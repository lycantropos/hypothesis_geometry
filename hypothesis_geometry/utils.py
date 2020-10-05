import math
from collections import deque
from functools import partial
from itertools import (cycle,
                       groupby)
from math import atan2
from operator import itemgetter
from random import Random
from typing import (Callable,
                    Iterable,
                    List,
                    Optional,
                    Sequence,
                    Tuple)

from dendroid import red_black
from dendroid.hints import Key
from locus import segmental
from robust.linear import (SegmentsRelationship,
                           segments_relationship)

from .core import triangular
from .core.contracts import (has_horizontal_lowermost_segment,
                             has_vertical_leftmost_segment)
from .core.subdivisional import (QuadEdge,
                                 to_edge_neighbours)
from .core.utils import (Orientation,
                         contour_to_centroid,
                         orientation,
                         point_in_angle,
                         points_to_centroid)
from .hints import (Contour,
                    Coordinate,
                    Domain,
                    Multicontour,
                    Multisegment,
                    Point,
                    Polygon,
                    Range,
                    Segment)

Chooser = Callable[[Sequence[Domain]], Domain]


def to_contour(points: Sequence[Point], size: int) -> Contour:
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
    candidates = red_black.set_(*filter(is_mouth, boundary_edges),
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


def to_star_contour(points: Sequence[Point]) -> Contour:
    centroid = points_to_centroid(points)
    result, prev_size = points, len(points) + 1
    while 2 < len(result) < prev_size:
        prev_size = len(result)
        centroid_x, centroid_y = centroid
        result = [deque(candidates,
                        maxlen=1)[0][1]
                  for _, candidates in groupby(sorted(
                    (_to_segment_angle(centroid_x, centroid_y, point), point)
                    for point in result),
                    key=itemgetter(0))]
        if len(result) > 2:
            centroid = contour_to_centroid(result)
            index = 0
            while max(index, 2) < len(result):
                if not point_in_angle(centroid,
                                      result[index - 1],
                                      result[index],
                                      result[(index + 1) % len(result)]):
                    del result[index]
                index += 1
            shrink_collinear_vertices(result)
    return result


def _to_segment_angle(start_x: Coordinate, start_y: Coordinate,
                      end: Point) -> Coordinate:
    end_x, end_y = end
    return math.atan2(end_y - start_y, end_x - start_x)


def to_multicontour(vertices: List[Point],
                    sizes: List[int],
                    chooser: Chooser) -> Multicontour:
    sorting_key_chooser = partial(chooser, [None, itemgetter(1, 0)])
    current_sorting_key = sorting_key_chooser()
    vertices = sorted(vertices,
                      key=current_sorting_key)
    predicates = cycle((has_vertical_leftmost_segment,
                        has_horizontal_lowermost_segment)
                       if current_sorting_key is None
                       else (has_horizontal_lowermost_segment,
                             has_vertical_leftmost_segment))
    current_predicate = next(predicates)
    result = []
    for size in sizes:
        contour = to_contour(vertices[:size], size)
        result.append(contour)
        can_touch_next_contour = current_predicate(
                contour_to_multisegment(contour))
        vertices = vertices[size - can_touch_next_contour:]
        new_sorting_key = sorting_key_chooser()
        if new_sorting_key is not current_sorting_key:
            (vertices, current_sorting_key,
             current_predicate) = (sorted(vertices,
                                          key=new_sorting_key),
                                   new_sorting_key,
                                   next(predicates))
    return result


def to_polygon(points: Sequence[Point],
               border_size: int,
               holes_sizes: List[int],
               chooser: Chooser) -> Polygon:
    triangulation = triangular.delaunay(points)
    boundary_edges = triangular.to_boundary_edges(triangulation)
    boundary_vertices = {edge.start for edge in boundary_edges}
    sorting_key_chooser = partial(chooser, [None, itemgetter(1, 0)])
    current_sorting_key = sorting_key_chooser()
    inner_points = sorted(set(points) - boundary_vertices,
                          key=current_sorting_key)
    predicates = cycle((has_vertical_leftmost_segment,
                        has_horizontal_lowermost_segment)
                       if current_sorting_key is None
                       else (has_horizontal_lowermost_segment,
                             has_vertical_leftmost_segment))
    current_predicate = next(predicates)
    holes = []
    holes_multisegment = []
    for hole_size in holes_sizes:
        hole_points = inner_points[:hole_size]
        hole = to_contour(hole_points, hole_size)[::-1]
        holes.append(hole)
        hole_multisegment = contour_to_multisegment(hole)
        holes_multisegment.extend(hole_multisegment)
        boundary_vertices.update(hole_points)
        can_touch_next_hole = current_predicate(hole_multisegment)
        inner_points = inner_points[hole_size - can_touch_next_hole:]
        next_sorting_key = sorting_key_chooser()
        if next_sorting_key is not current_sorting_key:
            (current_sorting_key, inner_points,
             current_predicate) = (next_sorting_key,
                                   sorted(inner_points,
                                          key=next_sorting_key),
                                   next(predicates))

    def to_segment_cross_or_overlap_detector(multisegment: Multisegment
                                             ) -> Callable[[Segment], bool]:
        return (
            (lambda segment, to_nearest_segment=(segmental.Tree(multisegment)
                                                 .nearest_segment)
             : segments_cross_or_overlap(to_nearest_segment(segment), segment))
            if multisegment
            else (lambda segment: False))

    def is_mouth(edge: QuadEdge,
                 cross_or_overlap_holes: Callable[[Segment], bool]
                 = to_segment_cross_or_overlap_detector(holes_multisegment)
                 ) -> bool:
        neighbour_end = edge.left_from_start.end
        return (neighbour_end not in boundary_vertices
                and not cross_or_overlap_holes((edge.start, neighbour_end))
                and not cross_or_overlap_holes((edge.end, neighbour_end)))

    def segments_cross_or_overlap(left: Segment, right: Segment) -> bool:
        relationship = segments_relationship(left, right)
        return (relationship is SegmentsRelationship.CROSS
                or relationship is SegmentsRelationship.OVERLAP)

    edges_neighbours = {edge: to_edge_neighbours(edge)
                        for edge in boundary_edges}
    candidates = red_black.set_(*filter(is_mouth, boundary_edges),
                                key=_edge_key)
    current_border_size = len(to_strict_convex_hull(points))
    while current_border_size < border_size:
        try:
            edge = candidates.popmax()
        except KeyError:
            break
        if not is_mouth(edge):
            continue
        current_border_size += 1
        boundary_vertices.add(edge.left_from_start.end)
        triangulation.delete(edge)
        for neighbour in edges_neighbours.pop(edge):
            edges_neighbours[neighbour] = to_edge_neighbours(neighbour)
            candidates.add(neighbour)
    border = [edge.start
              for edge in triangular.to_boundary_edges(triangulation)]
    shrink_collinear_vertices(border)
    return border, holes


def _edge_key(edge: QuadEdge) -> Key:
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

    def to_vector_angle(vector: Tuple[Coordinate, Coordinate]) -> Key:
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


def pack(function: Callable[..., Range]
         ) -> Callable[[Iterable[Domain]], Range]:
    return partial(apply, function)


def apply(function: Callable[..., Range],
          args: Iterable[Domain]) -> Range:
    return function(*args)


def sort_pair(pair: Sequence[Domain]) -> Tuple[Domain, Domain]:
    first, second = pair
    return (first, second) if first < second else (second, first)


def contour_to_multisegment(contour: Contour) -> Multisegment:
    return [(contour[index - 1], contour[index])
            for index in range(len(contour))]


def polygon_to_border_multisegment(polygon: Polygon) -> Multisegment:
    border, _ = polygon
    return contour_to_multisegment(border)
