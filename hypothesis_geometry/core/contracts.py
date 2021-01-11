from typing import Sequence

from bentley_ottmann.planar import segments_cross_or_overlap
from ground.base import get_context
from ground.hints import (Point,
                          Segment)

from hypothesis_geometry.hints import (Contour,
                                       Coordinate,
                                       Multisegment)
from .utils import (Orientation,
                    to_orientations)


def has_horizontal_lowermost_segment(segments: Sequence[Segment]) -> bool:
    lowermost_segment = max(segments,
                            key=segment_to_min_y)
    min_y = segment_to_min_y(lowermost_segment)
    return (is_segment_horizontal(lowermost_segment)
            or any(segment_to_min_y(segment) == min_y
                   and is_segment_horizontal(segment)
                   for segment in segments))


def has_vertical_leftmost_segment(segments: Sequence[Segment]) -> bool:
    leftmost_segment = max(segments,
                           key=segment_to_max_x)
    max_x = segment_to_max_x(leftmost_segment)
    return (is_segment_vertical(leftmost_segment)
            or any(segment_to_max_x(segment) == max_x
                   and is_segment_vertical(segment)
                   for segment in segments))


def segment_to_max_x(segment: Segment) -> Coordinate:
    return min(segment.start.x, segment.end.x)


def segment_to_min_y(segment: Segment) -> Coordinate:
    return min(segment.start.y, segment.end.y)


def is_contour_non_convex(contour: Contour) -> bool:
    orientations = to_orientations(contour)
    base_orientation = next(orientations)
    # orientation change means
    # that internal angle is greater than 180 degrees
    return any(orientation is not base_orientation
               for orientation in orientations)


def is_contour_strict(contour: Contour) -> bool:
    return all(orientation is not Orientation.COLLINEAR
               for orientation in to_orientations(contour))


def is_multisegment_valid(multisegment: Multisegment) -> bool:
    return not segments_cross_or_overlap(multisegment)


def is_point_inside_circumcircle(first_vertex: Point,
                                 second_vertex: Point,
                                 third_vertex: Point,
                                 point: Point) -> bool:
    context = get_context()
    return context.point_point_point_incircle_test(first_vertex, second_vertex,
                                                   third_vertex, point) > 0


def is_segment_horizontal(segment: Segment) -> bool:
    return segment.start.y == segment.end.y


def is_segment_vertical(segment: Segment) -> bool:
    return segment.start.x == segment.end.x


def points_do_not_lie_on_the_same_line(points: Sequence[Point]) -> bool:
    return any(orientation is not Orientation.COLLINEAR
               for orientation in to_orientations(points))
