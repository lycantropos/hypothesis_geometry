from hypothesis import given, strategies as st

from hypothesis_geometry.planar import boxes
from tests.hints import ScalarT
from tests.strategies import (
    data_object_strategy,
    scalar_strategy_strategy,
    scalar_strategy_with_limit_and_type_pair_strategy,
    scalar_strategy_with_limits_and_type_strategy,
)
from tests.utils import (
    ScalarStrategyLimitsWithType,
    box_has_coordinate_types,
    box_has_coordinates_in_range,
    context,
)


@given(scalar_strategy_strategy)
def test_basic(coordinates: st.SearchStrategy[ScalarT]) -> None:
    result = boxes(coordinates, context=context)

    assert isinstance(result, st.SearchStrategy)


@given(data_object_strategy, scalar_strategy_with_limit_and_type_pair_strategy)
def test_properties(
    data: st.DataObject,
    coordinates_limits_type_pair: tuple[
        ScalarStrategyLimitsWithType[ScalarT],
        ScalarStrategyLimitsWithType[ScalarT],
    ],
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

    strategy = boxes(x_coordinates, y_coordinates, context=context)

    result = data.draw(strategy)

    assert isinstance(result, context.box_cls)
    assert box_has_coordinate_types(result, x_type=x_type, y_type=y_type)
    assert box_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )
    assert result.min_x < result.max_x
    assert result.min_y < result.max_y


@given(data_object_strategy, scalar_strategy_with_limits_and_type_strategy)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = boxes(coordinates, context=context)

    result = data.draw(strategy)

    assert isinstance(result, context.box_cls)
    assert box_has_coordinate_types(result, x_type=type_, y_type=type_)
    assert box_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
    assert result.min_x < result.max_x
    assert result.min_y < result.max_y
