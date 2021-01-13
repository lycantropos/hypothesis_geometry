from enum import (IntEnum,
                  unique)


@unique
class MinContourSize(IntEnum):
    CONCAVE = 4
    CONVEX = 3


MIN_CONTOUR_SIZE = min(MinContourSize)
