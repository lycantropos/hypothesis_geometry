import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import HypothesisWarning

from hypothesis_geometry.planar import contours
from tests import strategies
from tests.hints import ScalarT
from tests.utils import (
    ScalarStrategyLimitsWithType,
    SizePair,
    are_vertices_strict,
    context,
    contour_has_coordinate_types,
    contour_has_coordinates_in_range,
    contour_has_valid_sizes,
    is_contour_counterclockwise,
    is_contour_non_self_intersecting,
    is_contour_strict,
)


@given(strategies.scalars_strategies, strategies.convex_contours_sizes_pairs)
def test_basic(
    coordinates: st.SearchStrategy[ScalarT], sizes_pair: SizePair
) -> None:
    min_size, max_size = sizes_pair

    result = contours(coordinates, min_size=min_size, max_size=max_size)

    assert isinstance(result, st.SearchStrategy)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategy_with_limit_and_type_pairs,
    strategies.convex_contours_sizes_pairs,
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

    strategy = contours(
        x_coordinates, y_coordinates, min_size=min_size, max_size=max_size
    )

    result = data.draw(strategy)

    assert isinstance(result, context.contour_cls)
    assert contour_has_valid_sizes(
        result, min_size=min_size, max_size=max_size
    )
    assert contour_has_coordinate_types(result, x_type=x_type, y_type=y_type)
    assert contour_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )
    assert is_contour_strict(result)
    assert is_contour_non_self_intersecting(result)
    assert is_contour_counterclockwise(result)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategies_with_limits_and_types,
    strategies.convex_contours_sizes_pairs,
)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
    sizes_pair: SizePair,
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair

    strategy = contours(coordinates, min_size=min_size, max_size=max_size)

    result = data.draw(strategy)

    assert isinstance(result, context.contour_cls)
    assert contour_has_valid_sizes(
        result, min_size=min_size, max_size=max_size
    )
    assert contour_has_coordinate_types(result, x_type=type_, y_type=type_)
    assert contour_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
    assert are_vertices_strict(result.vertices)
    assert is_contour_non_self_intersecting(result)
    assert is_contour_counterclockwise(result)


@given(
    strategies.scalars_strategies,
    strategies.invalid_convex_contours_sizes_pairs,
)
def test_invalid_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        contours(coordinates, min_size=min_size, max_size=max_size)


@given(
    strategies.scalars_strategies,
    strategies.non_valid_convex_contours_sizes_pairs,
)
def test_non_valid_sizes(
    coordinates: st.SearchStrategy[ScalarT], non_valid_sizes_pair: SizePair
) -> None:
    min_size, max_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        contours(coordinates, min_size=min_size, max_size=max_size)

    assert len(warnings) == 1
