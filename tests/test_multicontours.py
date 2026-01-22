import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import HypothesisWarning

from hypothesis_geometry.planar import multicontours
from tests.hints import ScalarT
from tests.strategies import (
    concave_contour_size_pair_strategy,
    convex_contour_size_pair_strategy,
    data_object_strategy,
    invalid_convex_contour_size_pair_strategy,
    invalid_multicontour_size_pair_strategy,
    multicontour_size_pair_strategy,
    non_valid_convex_contour_size_pair_strategy,
    scalar_strategy_strategy,
    scalar_strategy_with_limit_and_type_pair_strategy,
    scalar_strategy_with_limits_and_type_strategy,
)
from tests.utils import (
    ScalarStrategyLimitsWithType,
    SizePair,
    context,
    contours_do_not_cross_or_overlap,
    has_valid_size,
    is_contour_counterclockwise,
    is_contour_non_self_intersecting,
    is_multicontour,
    is_multicontour_strict,
    multicontour_has_coordinate_types,
    multicontour_has_coordinates_in_range,
    multicontour_has_valid_sizes,
)


@given(
    scalar_strategy_strategy,
    multicontour_size_pair_strategy,
    convex_contour_size_pair_strategy,
)
def test_basic(
    coordinates: st.SearchStrategy[ScalarT],
    sizes_pair: SizePair,
    contours_sizes_pair: SizePair,
) -> None:
    min_size, max_size = sizes_pair
    min_contour_size, max_contour_size = contours_sizes_pair

    result = multicontours(
        coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
        min_contour_size=min_contour_size,
        max_contour_size=max_contour_size,
    )

    assert isinstance(result, st.SearchStrategy)


@given(
    data_object_strategy,
    scalar_strategy_with_limit_and_type_pair_strategy,
    multicontour_size_pair_strategy,
    concave_contour_size_pair_strategy,
)
def test_properties(
    data: st.DataObject,
    coordinates_limits_type_pair: tuple[
        ScalarStrategyLimitsWithType[ScalarT],
        ScalarStrategyLimitsWithType[ScalarT],
    ],
    sizes_pair: SizePair,
    contours_sizes_pair: SizePair,
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
    min_contour_size, max_contour_size = contours_sizes_pair

    strategy = multicontours(
        x_coordinates,
        y_coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
        min_contour_size=min_contour_size,
        max_contour_size=max_contour_size,
    )

    result = data.draw(strategy)

    assert is_multicontour(result)
    assert multicontour_has_valid_sizes(
        result,
        min_size=min_size,
        max_size=max_size,
        min_contour_size=min_contour_size,
        max_contour_size=max_contour_size,
    )
    assert multicontour_has_coordinate_types(
        result, x_type=x_type, y_type=y_type
    )
    assert multicontour_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )
    assert is_multicontour_strict(result)
    assert all(is_contour_non_self_intersecting(contour) for contour in result)
    assert contours_do_not_cross_or_overlap(result)
    assert all(is_contour_counterclockwise(contour) for contour in result)


@given(
    data_object_strategy,
    scalar_strategy_with_limits_and_type_strategy,
    multicontour_size_pair_strategy,
    convex_contour_size_pair_strategy,
)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
    sizes_pair: SizePair,
    contours_sizes_pair: SizePair,
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair
    min_contour_size, max_contour_size = contours_sizes_pair

    strategy = multicontours(
        coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
        min_contour_size=min_contour_size,
        max_contour_size=max_contour_size,
    )

    result = data.draw(strategy)

    assert is_multicontour(result)
    assert has_valid_size(result, min_size=min_size, max_size=max_size)
    assert multicontour_has_valid_sizes(
        result,
        min_size=min_size,
        max_size=max_size,
        min_contour_size=min_contour_size,
        max_contour_size=max_contour_size,
    )
    assert multicontour_has_coordinate_types(
        result, x_type=type_, y_type=type_
    )
    assert multicontour_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
    assert is_multicontour_strict(result)
    assert all(is_contour_non_self_intersecting(contour) for contour in result)
    assert contours_do_not_cross_or_overlap(result)
    assert all(is_contour_counterclockwise(contour) for contour in result)


@given(scalar_strategy_strategy, invalid_multicontour_size_pair_strategy)
def test_invalid_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        multicontours(
            coordinates, context=context, min_size=min_size, max_size=max_size
        )


@given(scalar_strategy_strategy, invalid_convex_contour_size_pair_strategy)
def test_invalid_contours_sizes(
    coordinates: st.SearchStrategy[ScalarT],
    invalid_contours_sizes_pair: SizePair,
) -> None:
    min_contour_size, max_contour_size = invalid_contours_sizes_pair

    with pytest.raises(ValueError):
        multicontours(
            coordinates,
            context=context,
            min_contour_size=min_contour_size,
            max_contour_size=max_contour_size,
        )


@given(scalar_strategy_strategy, non_valid_convex_contour_size_pair_strategy)
def test_non_valid_contours_sizes(
    coordinates: st.SearchStrategy[ScalarT], non_valid_sizes_pair: SizePair
) -> None:
    min_size, max_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        multicontours(
            coordinates,
            context=context,
            min_contour_size=min_size,
            max_contour_size=max_size,
        )

    assert len(warnings) == 1
