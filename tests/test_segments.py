from hypothesis import given, strategies as st

from hypothesis_geometry.planar import segments
from tests.hints import ScalarT
from tests.strategies import (
    data_object_strategy,
    scalar_strategy_strategy,
    scalar_strategy_with_limit_and_type_pair_strategy,
    scalar_strategy_with_limits_and_type_strategy,
)
from tests.utils import (
    ScalarStrategyLimitsWithType,
    context,
    segment_has_coordinate_types,
    segment_has_coordinates_in_range,
)


@given(scalar_strategy_strategy)
def test_basic(coordinates: st.SearchStrategy[ScalarT]) -> None:
    result = segments(coordinates, context=context)

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

    strategy = segments(x_coordinates, y_coordinates, context=context)

    result = data.draw(strategy)

    assert isinstance(result, context.segment_cls)
    assert segment_has_coordinate_types(result, x_type=x_type, y_type=y_type)
    assert segment_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )
    assert result.start != result.end


@given(data_object_strategy, scalar_strategy_with_limits_and_type_strategy)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = segments(coordinates, context=context)

    result = data.draw(strategy)

    assert isinstance(result, context.segment_cls)
    assert segment_has_coordinate_types(result, x_type=type_, y_type=type_)
    assert segment_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
    assert result.start != result.end
