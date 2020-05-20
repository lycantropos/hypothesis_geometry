from typing import List

from bentley_ottmann.planar import segments_cross_or_overlap
from robust import cocircular

from hypothesis_geometry.hints import (Contour,
                                       Multisegment,
                                       Point)
from .utils import (Orientation,
                    to_orientations)


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


def points_do_not_lie_on_the_same_line(points: List[Point]) -> bool:
    return any(orientation is not Orientation.COLLINEAR
               for orientation in to_orientations(points))
