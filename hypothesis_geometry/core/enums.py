from enum import (IntEnum,
                  unique)


@unique
class Size(IntEnum):
    MIN_POLYLINE = 2
    TRIANGULAR_CONTOUR = 3
    RECTANGULAR_CONTOUR = 4
