from typing import List

from bentley_ottmann.planar import segments_cross_or_overlap
from robust import cocircular

from hypothesis_geometry.hints import (Contour,
                                       Coordinate,
                                       Multisegment,
                                       Point,
                                       Segment)
from .utils import (Orientation,
                    to_orientations)


def has_horizontal_lowermost_segment(multisegment: Multisegment) -> bool:
    lowermost_segment = max(multisegment,
                            key=segment_to_min_y)
    min_y = segment_to_min_y(lowermost_segment)
    return (is_segment_horizontal(lowermost_segment)
            or any(segment_to_min_y(segment) == min_y
                   and is_segment_horizontal(segment)
                   for segment in multisegment))


def has_vertical_leftmost_segment(multisegment: Multisegment) -> bool:
    leftmost_segment = max(multisegment,
                           key=segment_to_max_x)
    max_x = segment_to_max_x(leftmost_segment)
    return (is_segment_vertical(leftmost_segment)
            or any(segment_to_max_x(segment) == max_x
                   and is_segment_vertical(segment)
                   for segment in multisegment))


def segment_to_max_x(segment: Segment) -> Coordinate:
    (start_x, _), (end_x, _) = segment
    return min(start_x, end_x)


def segment_to_min_y(segment: Segment) -> Coordinate:
    (_, start_y), (_, end_y) = segment
    return min(start_y, end_y)


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
    return cocircular.determinant(first_vertex, second_vertex, third_vertex,
                                  point) > 0


def is_segment_horizontal(segment: Segment) -> bool:
    (_, start_y), (_, end_y) = segment
    return start_y == end_y


def is_segment_vertical(segment: Segment) -> bool:
    (start_x, _), (end_x, _) = segment
    return start_x == end_x


def points_do_not_lie_on_the_same_line(points: List[Point]) -> bool:
    return any(orientation is not Orientation.COLLINEAR
               for orientation in to_orientations(points))
