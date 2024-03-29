import numbers
from fractions import Fraction
from functools import partial
from itertools import repeat
from operator import ne
from typing import (Optional,
                    Tuple,
                    Type)

from ground.hints import Scalar
from hypothesis import strategies

from hypothesis_geometry.core.constants import (MIN_MIX_COMPONENTS_COUNT,
                                                MIN_MULTICONTOUR_SIZE,
                                                MIN_MULTIPOINT_SIZE,
                                                MIN_MULTIPOLYGON_SIZE,
                                                MIN_MULTISEGMENT_SIZE,
                                                MinContourSize)
from hypothesis_geometry.core.utils import sort_pair
from hypothesis_geometry.hints import Strategy
from tests.utils import (Limits,
                         SizesPair,
                         identity,
                         pack,
                         to_pairs)

data = strategies.data()

MAX_SIZE = 10
MAX_VALUE = 10 ** 15
MIN_VALUE = -MAX_VALUE
scalars_strategies_factories = {float: partial(strategies.floats,
                                               min_value=MIN_VALUE,
                                               max_value=MAX_VALUE,
                                               allow_nan=False,
                                               allow_infinity=False),
                                Fraction: strategies.fractions,
                                int: strategies.integers}
scalars_types = strategies.sampled_from(list(scalars_strategies_factories
                                             .keys()))
scalars_strategies = strategies.sampled_from(
        [factory() for factory in scalars_strategies_factories.values()]
)


def to_sizes_pairs(min_size: int, max_size: int = MAX_SIZE
                   ) -> Strategy[Tuple[int, Optional[int]]]:
    assert max_size <= MAX_SIZE
    sizes = strategies.integers(min_size, max_size)
    return (strategies.tuples(sizes, strategies.none())
            | strategies.tuples(sizes, sizes).map(sort_pair))


def to_non_valid_sizes_pairs(min_valid_size: int
                             ) -> Strategy[Tuple[int, Optional[int]]]:
    return (strategies.tuples(strategies.integers(0, min_valid_size - 1),
                              strategies.integers(min_valid_size))
            .filter(lambda sizes_pair: ne(*sizes_pair))
            .map(sort_pair))


def to_invalid_sizes_pairs(min_valid_size: int
                           ) -> Strategy[Tuple[int, Optional[int]]]:
    max_invalid_size = min_valid_size - 1
    invalid_sizes = strategies.integers(max_value=max_invalid_size)
    valid_sizes = strategies.integers(min_valid_size)
    return (strategies.tuples(strategies.integers(max_value=-1), valid_sizes)
            | strategies.tuples(invalid_sizes, invalid_sizes).map(sort_pair)
            | (strategies.tuples(valid_sizes, valid_sizes)
               .filter(lambda sizes_pair: ne(*sizes_pair))
               .map(sort_pair)
               .map(reversed)
               .map(tuple)))


concave_contours_sizes_pairs = to_sizes_pairs(MinContourSize.CONCAVE)
convex_contours_sizes_pairs = to_sizes_pairs(MinContourSize.CONVEX)


def is_valid_mix_components_sizes_pairs_triplet(
        sizes_pairs_triplet: Tuple[SizesPair, SizesPair, SizesPair]) -> bool:
    return (sum(component_max_size is None or bool(component_max_size)
                for _, component_max_size in sizes_pairs_triplet)
            >= MIN_MIX_COMPONENTS_COUNT)


mix_components_sizes_pairs_triplets = (
    (strategies.tuples(*repeat(to_sizes_pairs(0), 3))
     .filter(is_valid_mix_components_sizes_pairs_triplet))
)
multicontours_sizes_pairs = to_sizes_pairs(MIN_MULTICONTOUR_SIZE, 5)
multipoints_sizes_pairs = to_sizes_pairs(MIN_MULTIPOINT_SIZE)
multipolygons_sizes_pairs = to_sizes_pairs(MIN_MULTIPOLYGON_SIZE, 5)
multisegments_sizes_pairs = to_sizes_pairs(MIN_MULTISEGMENT_SIZE)
polygon_holes_sizes_pairs = to_sizes_pairs(0, 5)
non_valid_concave_contours_sizes_pairs = to_non_valid_sizes_pairs(
        MinContourSize.CONCAVE
)
non_valid_convex_contours_sizes_pairs = to_non_valid_sizes_pairs(
        MinContourSize.CONVEX
)
invalid_concave_contours_sizes_pairs = to_invalid_sizes_pairs(
        MinContourSize.CONCAVE
)
invalid_convex_contours_sizes_pairs = to_invalid_sizes_pairs(
        MinContourSize.CONVEX
)
invalid_mix_components_sizes_pairs_triplets = (
    (strategies.permutations([to_sizes_pairs(0), strategies.just((0, 0)),
                              strategies.just((0, 0))])
     .flatmap(pack(strategies.tuples)))
)
invalid_multicontours_sizes_pairs = to_invalid_sizes_pairs(
        MIN_MULTICONTOUR_SIZE
)
invalid_polygon_holes_sizes_pairs = to_invalid_sizes_pairs(0)
invalid_multipoints_sizes_pairs = to_invalid_sizes_pairs(MIN_MULTIPOINT_SIZE)
invalid_multipolygons_sizes_pairs = to_invalid_sizes_pairs(
        MIN_MULTIPOLYGON_SIZE
)
invalid_mix_points_sizes_pairs = to_invalid_sizes_pairs(0)
invalid_mix_polygons_sizes_pairs = to_invalid_sizes_pairs(0)
invalid_mix_segments_sizes_pairs = to_invalid_sizes_pairs(0)
invalid_multisegments_sizes_pairs = to_invalid_sizes_pairs(
        MIN_MULTISEGMENT_SIZE
)


def to_coordinates_strategies_with_limits_and_types(
        type_: Type[Scalar]
) -> Strategy[Tuple[Tuple[Strategy[Scalar], Limits], Type[Scalar]]]:
    strategy_factory = scalars_strategies_factories[type_]

    def to_strategy_with_limits(limits: Limits
                                ) -> Tuple[Strategy[Scalar], Limits]:
        min_value, max_value = limits
        return (strategy_factory(min_value=min_value,
                                 max_value=max_value),
                limits)

    def to_limits(coordinates: Strategy[Scalar]) -> Strategy[Limits]:
        result = strategies.tuples(coordinates, coordinates)
        result = result.filter(are_pair_coordinates_sparse
                               if issubclass(type_, numbers.Integral)
                               else pack(ne))
        result = result.map(sort_pair)
        return (strategies.tuples(coordinates, strategies.none()) | result
                if type_ is not float
                else result)

    return strategies.tuples(strategies.builds(to_strategy_with_limits,
                                               to_limits(strategy_factory())),
                             strategies.just(type_))


def are_pair_coordinates_sparse(pair: Tuple[Scalar, Scalar]) -> bool:
    first, second = pair
    return abs(first - second) >= 300


scalars_strategies_with_limits_and_types_strategies = (
    scalars_types.map(to_coordinates_strategies_with_limits_and_types)
)
scalars_strategies_with_limits_and_types = (
    scalars_strategies_with_limits_and_types_strategies.flatmap(identity)
)
scalars_strategy_with_limit_and_type_pairs = (
    scalars_strategies_with_limits_and_types_strategies.flatmap(to_pairs)
)
