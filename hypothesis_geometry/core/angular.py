from enum import (IntEnum,
                  unique)

from robust import parallelogram

from hypothesis_geometry.hints import (Point,
                                       Scalar)
from .utils import to_real_point


@unique
class Orientation(IntEnum):
    CLOCKWISE = -1
    COLLINEAR = 0
    COUNTERCLOCKWISE = 1


def to_orientation(first_ray_point: Point,
                   vertex: Point,
                   second_ray_point: Point) -> Orientation:
    return Orientation(to_sign(parallelogram.signed_area(
            to_real_point(vertex), to_real_point(first_ray_point),
            to_real_point(vertex), to_real_point(second_ray_point))))


def to_sign(value: Scalar) -> int:
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0
