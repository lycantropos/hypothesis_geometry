from enum import (IntEnum,
                  unique)


@unique
class MinContourSize(IntEnum):
    CONCAVE = 4
    CONVEX = 3


MIN_CONTOUR_SIZE = min(MinContourSize)
MIN_MULTIPOINT_SIZE = 1
MIN_MULTISEGMENT_SIZE = MIN_MULTIPOLYGON_SIZE = 2
