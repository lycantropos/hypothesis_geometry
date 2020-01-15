from typing import (Sequence,
                    Tuple)

from robust import cocircular

from hypothesis_geometry.hints import (Contour,
                                       Point)
from .utils import (Orientation,
                    to_orientation,
                    to_orientations,
                    to_real_point)

Segment = Tuple[Point, Point]


def is_point_inside_circumcircle(first_vertex: Point,
                                 second_vertex: Point,
                                 third_vertex: Point,
                                 point: Point) -> bool:
    return cocircular.determinant(to_real_point(first_vertex),
                                  to_real_point(second_vertex),
                                  to_real_point(third_vertex),
                                  to_real_point(point)) > 0


def is_contour_non_convex(contour: Contour) -> bool:
    orientations = to_orientations(contour)
    base_orientation = next(orientations)
    # orientation change means
    # that internal angle is greater than 180 degrees
    return not all(orientation == base_orientation
                   for orientation in orientations)


def points_do_not_lie_on_the_same_line(points: Sequence[Point]) -> bool:
    return any(orientation is not Orientation.COLLINEAR
               for orientation in to_orientations(points))


def is_contour_strict(contour: Contour) -> bool:
    return all(orientation is not Orientation.COLLINEAR
               for orientation in to_orientations(contour))


def is_non_self_intersecting_contour(contour: Contour) -> bool:
    edges = tuple((contour[index], contour[(index + 1) % len(contour)])
                  for index in range(len(contour)))
    for index, edge in enumerate(edges):
        # skipping neighbours because they always intersect
        # NOTE: first & last edges are neighbours
        if any(segments_intersect(edge, non_neighbour)
               for non_neighbour in _to_non_neighbours(index, edges)):
            return False
    return True


def _to_non_neighbours(edge_index: int,
                       edges: Sequence[Segment]) -> Sequence[Segment]:
    return (edges[max(edge_index + 2 - len(edges), 0):max(edge_index - 1, 0)]
            + edges[edge_index + 2:edge_index - 1 + len(edges)])


def segments_intersect(left: Segment, right: Segment) -> bool:
    if left == right:
        return True
    left_start, left_end = left
    self_start_orientation = _edge_orientation_with(right, left_start)
    self_end_orientation = _edge_orientation_with(right, left_end)
    if (self_start_orientation is Orientation.COLLINEAR
            and _in_interval(left_start, right)):
        return True
    elif (self_end_orientation is Orientation.COLLINEAR
          and _in_interval(left_end, right)):
        return True
    right_start, right_end = right
    other_start_orientation = _edge_orientation_with(left, right_start)
    other_end_orientation = _edge_orientation_with(left, right_end)
    if (self_start_orientation * self_end_orientation < 0
            and other_start_orientation * other_end_orientation < 0):
        return True
    elif (other_start_orientation is Orientation.COLLINEAR
          and _in_interval(right_start, left)):
        return True
    elif (other_end_orientation is Orientation.COLLINEAR
          and _in_interval(right_end, left)):
        return True
    else:
        return False


def _edge_orientation_with(edge: Segment, point: Point) -> Orientation:
    start, end = edge
    return to_orientation(end, start, point)


def _in_interval(point: Point, segment: Segment) -> bool:
    segment_start, segment_end = segment
    if point == segment_start:
        return True
    elif point == segment_end:
        return True
    else:
        segment_start_x, segment_start_y = segment_start
        segment_end_x, segment_end_y = segment_end
        left_x, right_x = ((segment_start_x, segment_end_x)
                           if segment_start_x < segment_end_x
                           else (segment_end_x, segment_start_x))
        bottom_y, top_y = ((segment_start_y, segment_end_y)
                           if segment_start_y < segment_end_y
                           else (segment_end_y, segment_start_y))
        point_x, point_y = point
        return left_x <= point_x <= right_x and bottom_y <= point_y <= top_y
