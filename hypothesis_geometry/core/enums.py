from enum import (IntEnum,
                  unique)


@unique
class Size(IntEnum):
    TRIANGULAR_CONTOUR = 3
    RECTANGULAR_CONTOUR = 4
