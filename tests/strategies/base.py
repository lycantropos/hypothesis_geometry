from decimal import Decimal
from fractions import Fraction
from functools import partial
from operator import ne
from typing import (Any,
                    Optional,
                    Tuple,
                    Type)

from hypothesis import strategies

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import (MIN_CONCAVE_CONTOUR_SIZE,
                                        MIN_POLYLINE_SIZE,
                                        TRIANGLE_SIZE)
from tests.utils import (Limits,
                         identity,
                         to_pairs)

data = strategies.data()

MAX_VALUE = 10 ** 15
MIN_VALUE = -MAX_VALUE
coordinates_strategies_factories = {Decimal: partial(strategies.decimals,
                                                     min_value=MIN_VALUE,
                                                     max_value=MAX_VALUE,
                                                     allow_nan=False,
                                                     allow_infinity=False),
                                    float: partial(strategies.floats,
                                                   min_value=MIN_VALUE,
                                                   max_value=MAX_VALUE,
                                                   allow_nan=False,
                                                   allow_infinity=False),
                                    Fraction: strategies.fractions,
                                    int: strategies.integers}
coordinates_types = strategies.sampled_from(
        list(coordinates_strategies_factories.keys()))
coordinates_strategies = strategies.sampled_from(
        [factory() for factory in coordinates_strategies_factories.values()])


def to_sizes_pairs(min_size: int) -> Strategy[Tuple[int, Optional[int]]]:
    sizes = strategies.integers(min_size, 30)
    return (strategies.tuples(sizes, strategies.none())
            | strategies.tuples(sizes, sizes).map(sort_pair))


def to_invalid_sizes_pairs(max_invalid_size: int
                           ) -> Strategy[Tuple[int, Optional[int]]]:
    invalid_sizes = strategies.integers(max_value=max_invalid_size)
    valid_sizes = strategies.integers(max_invalid_size)
    return ((strategies.tuples(valid_sizes, valid_sizes)
             .filter(lambda sizes_pair: ne(*sizes_pair))
             .map(sort_pair)
             .map(reversed)
             .map(tuple))
            | strategies.tuples(invalid_sizes, invalid_sizes).map(sort_pair))


def sort_pair(pair: Tuple[Any, Any]) -> Tuple[Any, Any]:
    first, second = pair
    return (first, second) if first < second else (second, first)


concave_contours_sizes_pairs = to_sizes_pairs(MIN_CONCAVE_CONTOUR_SIZE)
convex_contours_sizes_pairs = to_sizes_pairs(TRIANGLE_SIZE)
polylines_sizes_pairs = to_sizes_pairs(MIN_POLYLINE_SIZE)
invalid_concave_contours_sizes_pairs = to_invalid_sizes_pairs(TRIANGLE_SIZE)
invalid_convex_contours_sizes_pairs = to_invalid_sizes_pairs(TRIANGLE_SIZE - 1)
invalid_polylines_sizes_pairs = to_invalid_sizes_pairs(MIN_POLYLINE_SIZE - 1)


def to_coordinates_strategies_with_limits_and_types(
        type_: Type[Coordinate]) -> Strategy[Tuple[Tuple[Strategy[Coordinate],
                                                         Limits],
                                                   Type[Coordinate]]]:
    strategy_factory = coordinates_strategies_factories[type_]

    def to_strategy_with_limits(limits: Limits
                                ) -> Tuple[Strategy[Coordinate], Limits]:
        min_value, max_value = limits
        return (strategy_factory(min_value=min_value,
                                 max_value=max_value),
                limits)

    def to_limits(coordinates: Strategy[Coordinate]) -> Strategy[Limits]:
        result = (strategies.tuples(coordinates, coordinates)
                  .filter(are_pair_coordinates_sparse)
                  .map(sort_pair))
        return (strategies.tuples(coordinates, strategies.none()) | result
                if type_ is not float and type_ is not Decimal
                else result)

    return strategies.tuples(strategies.builds(to_strategy_with_limits,
                                               to_limits(strategy_factory())),
                             strategies.just(type_))


def are_pair_coordinates_sparse(pair: Tuple[Coordinate, Coordinate]) -> bool:
    first, second = pair
    return abs(first - second) >= 10


coordinates_strategies_with_limits_and_types_strategies = (
    coordinates_types.map(to_coordinates_strategies_with_limits_and_types))
coordinates_strategies_with_limits_and_types = (
    coordinates_strategies_with_limits_and_types_strategies.flatmap(identity))
coordinates_strategy_with_limit_and_type_pairs = (
    coordinates_strategies_with_limits_and_types_strategies.flatmap(to_pairs))
