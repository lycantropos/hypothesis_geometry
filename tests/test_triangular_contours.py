from hypothesis import given, strategies as st

from hypothesis_geometry.planar import triangular_contours
from tests import strategies
from tests.hints import ScalarT
from tests.utils import (
    ScalarStrategyLimitsWithType,
    are_vertices_non_convex,
    contour_has_coordinate_types,
    contour_has_coordinates_in_range,
    contour_has_valid_sizes,
    is_contour,
    is_contour_counterclockwise,
    is_contour_non_self_intersecting,
    is_contour_strict,
)


@given(strategies.scalars_strategies)
def test_basic(coordinates: st.SearchStrategy[ScalarT]) -> None:
    result = triangular_contours(coordinates)

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

    strategy = triangular_contours(x_coordinates, y_coordinates)

    result = data.draw(strategy)

    assert is_contour(result)
    assert contour_has_valid_sizes(result, min_size=3, max_size=3)
    assert contour_has_coordinate_types(result, x_type=x_type, y_type=y_type)
    assert contour_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )
    assert is_contour_strict(result)
    assert not are_vertices_non_convex(result.vertices)
    assert is_contour_non_self_intersecting(result)
    assert is_contour_counterclockwise(result)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategies_with_limits_and_types,
)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = triangular_contours(coordinates)

    result = data.draw(strategy)

    assert is_contour(result)
    assert contour_has_valid_sizes(result, min_size=3, max_size=3)
    assert contour_has_coordinate_types(result, x_type=type_, y_type=type_)
    assert contour_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
    assert is_contour_strict(result)
    assert not are_vertices_non_convex(result.vertices)
    assert is_contour_non_self_intersecting(result)
    assert is_contour_counterclockwise(result)
