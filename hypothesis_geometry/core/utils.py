from itertools import chain
from typing import (Iterable,
                    Iterator,
                    Sequence,
                    Tuple)

from ground.base import (Orientation,
                         get_context)
from ground.hints import Point

from hypothesis_geometry.hints import (Contour,
                                       Domain)

flatten = chain.from_iterable


def pairwise(iterable: Iterable[Domain]) -> Iterable[Tuple[Domain, Domain]]:
    iterator = iter(iterable)
    element = next(iterator, None)
    for next_element in iterator:
        yield element, next_element
        element = next_element


Orientation = Orientation


def orientation(first, vertex, second):
    context = get_context()
    return context.angle_orientation(vertex, first, second)


def to_orientations(contour: Contour) -> Iterator[Orientation]:
    return (orientation(contour[index - 1], contour[index],
                        contour[(index + 1) % len(contour)])
            for index in range(len(contour)))


def points_to_centroid(points: Sequence[Point]) -> Point:
    return get_context().multipoint_centroid(points)


def contour_to_centroid(contour: Contour) -> Point:
    return get_context().contour_centroid(contour)


def point_in_angle(point: Point,
                   first_ray_point: Point,
                   vertex: Point,
                   second_ray_point: Point) -> bool:
    angle_orientation = orientation(first_ray_point, vertex, second_ray_point)
    first_half_orientation = orientation(first_ray_point, vertex, point)
    second_half_orientation = orientation(vertex, second_ray_point, point)
    return (second_half_orientation is angle_orientation
            if first_half_orientation is Orientation.COLLINEAR
            else (first_half_orientation is angle_orientation
                  if second_half_orientation is Orientation.COLLINEAR
                  else (first_half_orientation is second_half_orientation
                        is (angle_orientation
                            # if angle is degenerate
                            or Orientation.COUNTERCLOCKWISE))))
