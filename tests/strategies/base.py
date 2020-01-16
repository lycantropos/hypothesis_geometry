from decimal import Decimal
from fractions import Fraction
from functools import partial
from typing import (Any,
                    Tuple,
                    Type)

from hypothesis import strategies

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from tests.utils import Limits

data = strategies.data()

coordinates_strategies_factories = {Decimal: partial(strategies.decimals,
                                                     allow_nan=False,
                                                     allow_infinity=False),
                                    float: partial(strategies.floats,
                                                   allow_nan=False,
                                                   allow_infinity=False),
                                    Fraction: strategies.fractions,
                                    int: strategies.integers}
coordinates_types = strategies.sampled_from(
        list(coordinates_strategies_factories.keys()))
coordinates_strategies = strategies.sampled_from(
        [factory() for factory in coordinates_strategies_factories.values()])


def to_coordinates_strategies_with_limits_and_types(
        type_: Type[Coordinate]) -> Strategy[Tuple[Tuple[Strategy[Coordinate],
                                                         Limits],
                                                   Type[Coordinate]]]:
    def to_strategy_with_limits(limits: Limits
                                ) -> Tuple[Strategy[Coordinate], Limits]:
        min_value, max_value = limits
        return (strategy_factory(min_value=min_value,
                                 max_value=max_value),
                limits)

    def to_limits(coordinates: Strategy[Coordinate]) -> Limits:
        return (strategies.tuples(coordinates, strategies.none())
                | (strategies.tuples(coordinates, coordinates)
                   .filter(are_pair_coordinates_sparse)
                   .map(sort_pair)))

    strategy_factory = coordinates_strategies_factories[type_]
    return strategies.tuples(strategies.builds(to_strategy_with_limits,
                                               to_limits(strategy_factory())),
                             strategies.just(type_))


def are_pair_coordinates_sparse(pair: Tuple[Any, Any]) -> bool:
    first, second = pair
    return abs(first - second) >= 10


def sort_pair(pair: Tuple[Any, Any]) -> Tuple[Any, Any]:
    first, second = pair
    return (first, second) if first < second else (second, first)


coordinates_strategies_with_limits_and_types = coordinates_types.flatmap(
        to_coordinates_strategies_with_limits_and_types)
