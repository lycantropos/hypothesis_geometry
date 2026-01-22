import pytest
from hypothesis import given, strategies as st

from hypothesis_geometry.planar import multipoints
from tests.hints import ScalarT
from tests.strategies import (
    data_object_strategy,
    invalid_multipoint_size_pair_strategy,
    multipoint_size_pair_strategy,
    scalar_strategy_strategy,
    scalar_strategy_with_limit_and_type_pair_strategy,
    scalar_strategy_with_limits_and_type_strategy,
)
from tests.utils import (
    ScalarStrategyLimitsWithType,
    SizePair,
    all_unique,
    context,
    has_valid_size,
    multipoint_has_coordinate_types,
    multipoint_has_coordinates_in_range,
)


@given(scalar_strategy_strategy, multipoint_size_pair_strategy)
def test_basic(
    coordinates: st.SearchStrategy[ScalarT], sizes_pair: SizePair
) -> None:
    min_size, max_size = sizes_pair

    result = multipoints(
        coordinates, context=context, min_size=min_size, max_size=max_size
    )

    assert isinstance(result, st.SearchStrategy)


@given(
    data_object_strategy,
    scalar_strategy_with_limit_and_type_pair_strategy,
    multipoint_size_pair_strategy,
)
def test_properties(
    data: st.DataObject,
    coordinates_limits_type_pair: tuple[
        ScalarStrategyLimitsWithType[ScalarT],
        ScalarStrategyLimitsWithType[ScalarT],
    ],
    sizes_pair: SizePair,
) -> None:
    (x_coordinates_limits_type, y_coordinates_limits_type) = (
        coordinates_limits_type_pair
    )
    ((x_coordinates, (min_x_value, max_x_value)), x_type) = (
        x_coordinates_limits_type
    )
    ((y_coordinates, (min_y_value, max_y_value)), y_type) = (
        y_coordinates_limits_type
    )
    min_size, max_size = sizes_pair

    strategy = multipoints(
        x_coordinates,
        y_coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
    )

    result = data.draw(strategy)

    assert isinstance(result, context.multipoint_cls)
    assert has_valid_size(result.points, min_size=min_size, max_size=max_size)
    assert multipoint_has_coordinate_types(
        result, x_type=x_type, y_type=y_type
    )
    assert multipoint_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )
    assert all_unique(result.points)


@given(
    data_object_strategy,
    scalar_strategy_with_limits_and_type_strategy,
    multipoint_size_pair_strategy,
)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
    sizes_pair: SizePair,
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair

    strategy = multipoints(
        coordinates, context=context, min_size=min_size, max_size=max_size
    )

    result = data.draw(strategy)

    assert isinstance(result, context.multipoint_cls)
    assert has_valid_size(result.points, min_size=min_size, max_size=max_size)
    assert multipoint_has_coordinate_types(result, x_type=type_, y_type=type_)
    assert multipoint_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
    assert all_unique(result.points)


@given(scalar_strategy_strategy, invalid_multipoint_size_pair_strategy)
def test_invalid_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        multipoints(
            coordinates, context=context, min_size=min_size, max_size=max_size
        )
