from typing import (Iterable,
                    Optional,
                    Sequence,
                    Sized)

from bentley_ottmann.planar import segments_cross_or_overlap
from ground.base import (Context,
                         Orientation)
from ground.hints import (Point,
                          Scalar,
                          Segment)

from .hints import (Multicontour,
                    Orienteer)


def are_segments_non_crossing_non_overlapping(segments: Sequence[Segment],
                                              context: Context) -> bool:
    return not segments_cross_or_overlap(segments,
                                         context=context)


def has_valid_size(sized: Sized,
                   *,
                   min_size: int,
                   max_size: Optional[int]) -> bool:
    size = len(sized)
    return min_size <= size and (max_size is None or size <= max_size)


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


def is_segment_horizontal(segment: Segment) -> bool:
    return segment.start.y == segment.end.y


def is_segment_vertical(segment: Segment) -> bool:
    return segment.start.x == segment.end.x


def multicontour_has_valid_sizes(multicontour: Multicontour,
                                 *,
                                 min_size: int,
                                 max_size: Optional[int],
                                 min_contour_size: int,
                                 max_contour_size: Optional[int]) -> bool:
    return (has_valid_size(multicontour,
                           min_size=min_size,
                           max_size=max_size)
            and all(has_valid_size(contour.vertices,
                                   min_size=min_contour_size,
                                   max_size=max_contour_size)
                    for contour in multicontour))


def segment_to_max_x(segment: Segment[Scalar]) -> Scalar:
    return min(segment.start.x, segment.end.x)


def segment_to_min_y(segment: Segment[Scalar]) -> Scalar:
    return min(segment.start.y, segment.end.y)


def angle_contains_point(vertex: Point[Scalar],
                         first_ray_point: Point[Scalar],
                         second_ray_point: Point[Scalar],
                         point: Point[Scalar],
                         orienteer: Orienteer) -> bool:
    angle_orientation = orienteer(vertex, first_ray_point, second_ray_point)
    first_half_orientation = orienteer(vertex, first_ray_point, point)
    second_half_orientation = orienteer(second_ray_point, vertex, point)
    return (second_half_orientation is angle_orientation
            if first_half_orientation is Orientation.COLLINEAR
            else (first_half_orientation is angle_orientation
                  if second_half_orientation is Orientation.COLLINEAR
                  else (first_half_orientation is second_half_orientation
                        is (angle_orientation
                            # if angle is degenerate
                            or Orientation.COUNTERCLOCKWISE))))


def are_points_non_collinear(points: Sequence[Point[Scalar]],
                             orienteer: Orienteer) -> bool:
    return any(orientation is not Orientation.COLLINEAR
               for orientation in to_contour_orientations(points, orienteer))


def are_vertices_non_convex(vertices: Sequence[Point[Scalar]],
                            orienteer: Orienteer) -> bool:
    orientations = iter(to_contour_orientations(vertices, orienteer))
    base_orientation = next(orientations)
    # orientation change means
    # that internal angle is greater than 180 degrees
    return any(orientation is not base_orientation
               for orientation in orientations)


def are_vertices_strict(vertices: Sequence[Point[Scalar]],
                        orienteer: Orienteer) -> bool:
    return all(orientation is not Orientation.COLLINEAR
               for orientation in to_contour_orientations(vertices, orienteer))


def to_contour_orientations(vertices: Sequence[Point[Scalar]],
                            orienteer: Orienteer) -> Iterable[Orientation]:
    return (orienteer(vertices[index], vertices[index - 1],
                      vertices[(index + 1) % len(vertices)])
            for index in range(len(vertices)))
