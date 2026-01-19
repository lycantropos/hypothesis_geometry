from hypothesis import given, strategies as st

from hypothesis_geometry.planar import points
from tests import strategies
from tests.hints import ScalarT
from tests.utils import (
    ScalarStrategyLimitsWithType,
    context,
    point_has_coordinate_types,
    point_has_coordinates_in_range,
)


@given(strategies.scalars_strategies)
def test_basic(coordinates: st.SearchStrategy[ScalarT]) -> None:
    result = points(coordinates)

    assert isinstance(result, st.SearchStrategy)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategy_with_limit_and_type_pairs,
)
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

    strategy = points(x_coordinates, y_coordinates)

    result = data.draw(strategy)

    assert isinstance(result, context.point_cls)
    assert point_has_coordinate_types(result, x_type=x_type, y_type=y_type)
    assert point_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategies_with_limits_and_types,
)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = points(coordinates)

    result = data.draw(strategy)

    assert isinstance(result, context.point_cls)
    assert point_has_coordinate_types(result, x_type=type_, y_type=type_)
    assert point_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
