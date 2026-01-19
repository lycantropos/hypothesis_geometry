import numbers
from collections.abc import Callable
from fractions import Fraction
from functools import partial
from itertools import repeat
from operator import ne
from typing import Any, cast

from hypothesis import strategies as st

from hypothesis_geometry._core.constants import (
    MIN_MIX_COMPONENTS_COUNT,
    MIN_MULTICONTOUR_SIZE,
    MIN_MULTIPOINT_SIZE,
    MIN_MULTIPOLYGON_SIZE,
    MIN_MULTISEGMENT_SIZE,
    MinContourSize,
)
from hypothesis_geometry._core.utils import sort_pair
from tests.hints import ScalarT
from tests.utils import Limits, SizePair, context, identity, pack, to_pairs

data_object_strategy = st.data()

MAX_SIZE = 10


def scalar_type_to_strategy_factory(
    scalar_type: type[ScalarT], /
) -> Callable[..., st.SearchStrategy[ScalarT]]:
    if scalar_type is Fraction:
        return cast(Callable[..., st.SearchStrategy[ScalarT]], st.fractions)
    raise TypeError(scalar_type)


scalar_types = [Fraction]
scalar_type_strategy = st.sampled_from(scalar_types)
scalar_strategy_strategy: st.SearchStrategy[st.SearchStrategy[Any]] = (
    st.sampled_from(
        [
            scalar_type_to_strategy_factory(scalar_type)()
            for scalar_type in scalar_types
        ]
    )
)


def to_sizes_pairs(
    min_size: int, max_size: int = MAX_SIZE, /
) -> st.SearchStrategy[tuple[int, int | None]]:
    assert max_size <= MAX_SIZE
    sizes = st.integers(min_size, max_size)
    return st.tuples(sizes, st.none()) | st.tuples(sizes, sizes).map(sort_pair)


def to_non_valid_sizes_pairs(
    min_valid_size: int, /
) -> st.SearchStrategy[tuple[int, int | None]]:
    return (
        st.tuples(
            st.integers(0, min_valid_size - 1), st.integers(min_valid_size)
        )
        .filter(lambda sizes_pair: ne(*sizes_pair))
        .map(sort_pair)
    )


def to_invalid_sizes_pairs(
    min_valid_size: int, /
) -> st.SearchStrategy[tuple[int, int | None]]:
    max_invalid_size = min_valid_size - 1
    invalid_sizes = st.integers(max_value=max_invalid_size)
    valid_sizes = st.integers(min_valid_size)
    return (
        st.tuples(st.integers(max_value=-1), valid_sizes)
        | st.tuples(invalid_sizes, invalid_sizes).map(sort_pair)
        | (
            st.tuples(valid_sizes, valid_sizes)
            .filter(lambda sizes_pair: ne(*sizes_pair))
            .map(partial(sort_pair, reverse=True))
        )
    )


concave_contours_sizes_pairs = to_sizes_pairs(MinContourSize.CONCAVE)
convex_contours_sizes_pairs = to_sizes_pairs(MinContourSize.CONVEX)


def is_valid_mix_components_sizes_pairs_triplet(
    sizes_pairs_triplet: tuple[SizePair, SizePair, SizePair], /
) -> bool:
    return (
        sum(
            component_max_size is None or bool(component_max_size)
            for _, component_max_size in sizes_pairs_triplet
        )
        >= MIN_MIX_COMPONENTS_COUNT
    )


mix_components_sizes_pairs_triplets = st.tuples(
    *repeat(to_sizes_pairs(0), 3)
).filter(is_valid_mix_components_sizes_pairs_triplet)
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
invalid_mix_components_sizes_pairs_triplets = st.permutations(
    [to_sizes_pairs(0), st.just((0, 0)), st.just((0, 0))]
).flatmap(pack(st.tuples))
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


def to_coordinate_strategy_with_limit_and_type_strategy(
    scalar_type: type[ScalarT], /
) -> st.SearchStrategy[
    tuple[tuple[st.SearchStrategy[ScalarT], Limits[ScalarT]], type[ScalarT]]
]:
    strategy_factory = scalar_type_to_strategy_factory(scalar_type)

    def to_strategy_with_limits(
        limits: Limits[ScalarT], /
    ) -> tuple[st.SearchStrategy[ScalarT], Limits[ScalarT]]:
        min_value, max_value = limits
        return (
            strategy_factory(min_value=min_value, max_value=max_value),
            limits,
        )

    def to_limits(
        coordinates: st.SearchStrategy[ScalarT], /
    ) -> st.SearchStrategy[Limits[ScalarT]]:
        result = st.tuples(coordinates, coordinates)
        result = result.filter(
            are_pair_coordinates_sparse
            if issubclass(scalar_type, numbers.Integral)
            else pack(ne)
        )
        result = result.map(sort_pair)
        return (
            st.tuples(coordinates, st.none()) | result
            if scalar_type is not float
            else result
        )

    return st.tuples(
        st.builds(to_strategy_with_limits, to_limits(strategy_factory())),
        st.just(scalar_type),
    )


def are_pair_coordinates_sparse(pair: tuple[ScalarT, ScalarT], /) -> bool:
    first, second = pair
    return bool(context.coordinate_factory(300) <= abs(first - second))


scalars_strategies_with_limits_and_types_strategies: st.SearchStrategy[
    st.SearchStrategy[
        tuple[tuple[st.SearchStrategy[Any], Limits[Any]], type[Any]]
    ]
] = scalar_type_strategy.map(
    to_coordinate_strategy_with_limit_and_type_strategy
)
scalars_strategies_with_limits_and_types = (
    scalars_strategies_with_limits_and_types_strategies.flatmap(identity)
)
scalars_strategy_with_limit_and_type_pairs = (
    scalars_strategies_with_limits_and_types_strategies.flatmap(to_pairs)
)
