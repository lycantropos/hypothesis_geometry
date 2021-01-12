from functools import partial
from typing import (Callable,
                    Iterable,
                    Sequence)

from bentley_ottmann.planar import segments_cross_or_overlap
from ground.base import (Context,
                         Orientation,
                         get_context)
from ground.hints import (Coordinate,
                          Point,
                          Segment)

from .hints import (Orienteer,
                    QuaternaryPointFunction)
from .utils import to_contour_orienteer


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


def to_non_convex_vertices_detector(context: Context
                                    ) -> Callable[[Sequence[Point]], bool]:
    return partial(_are_vertices_non_convex, to_contour_orienteer(context))


def _are_vertices_non_convex(contour_orienteer
                             : Callable[[Sequence[Point]],
                                        Iterable[Orientation]],
                             vertices: Sequence[Point]) -> bool:
    orientations = iter(contour_orienteer(vertices))
    base_orientation = next(orientations)
    # orientation change means
    # that internal angle is greater than 180 degrees
    return any(orientation is not base_orientation
               for orientation in orientations)


def to_strict_vertices_detector(context: Context
                                ) -> Callable[[Sequence[Point]], bool]:
    return partial(_are_vertices_strict, to_contour_orienteer(context))


def _are_vertices_strict(contour_orienteer
                         : Callable[[Sequence[Point]], Iterable[Orientation]],
                         vertices: Sequence[Point]) -> bool:
    return all(orientation is not Orientation.COLLINEAR
               for orientation in contour_orienteer(vertices))


def are_segments_non_crossing_non_overlapping(segments: Sequence[Segment]
                                              ) -> bool:
    return not segments_cross_or_overlap(segments)


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


def to_non_collinear_points_detector(context: Context
                                     ) -> Callable[[Sequence[Point]], bool]:
    return partial(_are_points_non_collinear, to_contour_orienteer(context))


def _are_points_non_collinear(contour_orienteer
                              : Callable[[Sequence[Point]],
                                         Iterable[Orientation]],
                              points: Sequence[Point]) -> bool:
    return any(orientation is not Orientation.COLLINEAR
               for orientation in contour_orienteer(points))


def to_angle_containment_detector(context: Context
                                  ) -> QuaternaryPointFunction[bool]:
    return partial(_angle_contains_point, context.angle_orientation)


def _angle_contains_point(orienteer: Orienteer,
                          vertex: Point,
                          first_ray_point: Point,
                          second_ray_point: Point,
                          point: Point) -> bool:
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
