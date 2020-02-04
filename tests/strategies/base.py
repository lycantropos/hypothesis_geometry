from decimal import Decimal
from fractions import Fraction
from functools import partial
from operator import ne
from typing import (Any,
                    Tuple,
                    Type)

from hypothesis import strategies

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import TRIANGLE_SIZE
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

sizes = strategies.integers(TRIANGLE_SIZE, 100)
invalid_sizes = strategies.integers(max_value=TRIANGLE_SIZE - 1)


def sort_pair(pair: Tuple[Any, Any]) -> Tuple[Any, Any]:
    first, second = pair
    return (first, second) if first < second else (second, first)


sizes_pairs = (strategies.tuples(sizes, strategies.none())
               | strategies.tuples(sizes, sizes).map(sort_pair))
invalid_sizes_pairs = ((strategies.tuples(sizes, sizes)
                        .filter(lambda sizes_pair: ne(*sizes_pair))
                        .map(sort_pair)
                        .map(reversed)
                        .map(tuple))
                       | (strategies.tuples(invalid_sizes, invalid_sizes)
                          .map(sort_pair)))
coordinates_strategies_with_sizes_pairs = strategies.tuples(
        coordinates_strategies, sizes_pairs)


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
coordinates_strategies_limits_types_with_sizes_pairs = strategies.tuples(
        coordinates_strategies_with_limits_and_types, sizes_pairs)
coordinates_strategy_limits_type_pairs_with_sizes_pairs = strategies.tuples(
        coordinates_strategy_with_limit_and_type_pairs, sizes_pairs)
coordinates_strategies_with_invalid_sizes_pairs = strategies.tuples(
        coordinates_strategies, invalid_sizes_pairs)
