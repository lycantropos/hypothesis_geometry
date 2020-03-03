from typing import Sequence

from bentley_ottmann.planar import edges_intersect
from robust import cocircular

from hypothesis_geometry.hints import (Contour,
                                       Point)
from .utils import (Orientation,
                    _is_real_point,
                    _to_real_point,
                    to_orientations)


def is_point_inside_circumcircle(first_vertex: Point,
                                 second_vertex: Point,
                                 third_vertex: Point,
                                 point: Point) -> bool:
    if not _is_real_point(point):
        first_vertex, second_vertex, third_vertex, point = (
            _to_real_point(first_vertex), _to_real_point(second_vertex),
            _to_real_point(third_vertex), _to_real_point(point))
    return cocircular.determinant(first_vertex, second_vertex, third_vertex,
                                  point) > 0


def is_contour_non_convex(contour: Contour) -> bool:
    orientations = to_orientations(contour)
    base_orientation = next(orientations)
    # orientation change means
    # that internal angle is greater than 180 degrees
    return any(orientation is not base_orientation
               for orientation in orientations)


def points_do_not_lie_on_the_same_line(points: Sequence[Point]) -> bool:
    return any(orientation is not Orientation.COLLINEAR
               for orientation in to_orientations(points))


def is_contour_strict(contour: Contour) -> bool:
    return all(orientation is not Orientation.COLLINEAR
               for orientation in to_orientations(contour))


def is_non_self_intersecting_contour(contour: Contour) -> bool:
    return not edges_intersect(contour,
                               accurate=False)
