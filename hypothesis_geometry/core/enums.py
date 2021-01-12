from enum import IntEnum


class Size(IntEnum):
    EMPTY_SEQUENCE = 0
    MIN_CONCAVE_CONTOUR = 4
    MIN_POLYLINE = 2
    RECTANGULAR_CONTOUR = 4
    TRIANGULAR_CONTOUR = 3
