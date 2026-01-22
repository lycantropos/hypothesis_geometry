import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import HypothesisWarning

from hypothesis_geometry.planar import polygons
from tests import strategies
from tests.hints import ScalarT
from tests.utils import (
    ScalarStrategyLimitsWithType,
    SizePair,
    context,
    contours_do_not_cross_or_overlap,
    is_contour_counterclockwise,
    is_contour_non_self_intersecting,
    is_polygon_strict,
    polygon_has_coordinate_types,
    polygon_has_coordinates_in_range,
    polygon_has_valid_sizes,
)


@given(
    strategies.scalars_strategies,
    strategies.concave_contours_sizes_pairs,
    strategies.polygon_holes_sizes_pairs,
    strategies.convex_contours_sizes_pairs,
)
def test_basic(
    coordinates: st.SearchStrategy[ScalarT],
    sizes_pair: SizePair,
    holes_sizes_pair: SizePair,
    hole_sizes_pair: SizePair,
) -> None:
    min_size, max_size = sizes_pair
    min_holes_size, max_holes_size = holes_sizes_pair
    min_hole_size, max_hole_size = hole_sizes_pair

    result = polygons(
        coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )

    assert isinstance(result, st.SearchStrategy)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategy_with_limit_and_type_pairs,
    strategies.concave_contours_sizes_pairs,
    strategies.polygon_holes_sizes_pairs,
    strategies.convex_contours_sizes_pairs,
)
def test_properties(
    data: st.DataObject,
    coordinates_limits_type_pair: tuple[
        ScalarStrategyLimitsWithType[ScalarT],
        ScalarStrategyLimitsWithType[ScalarT],
    ],
    sizes_pair: SizePair,
    holes_sizes_pair: SizePair,
    hole_sizes_pair: SizePair,
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
    min_holes_size, max_holes_size = holes_sizes_pair
    min_hole_size, max_hole_size = hole_sizes_pair

    strategy = polygons(
        x_coordinates,
        y_coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )

    result = data.draw(strategy)

    assert isinstance(result, context.polygon_cls)
    assert polygon_has_valid_sizes(
        result,
        min_size=min_size,
        max_size=max_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )
    assert polygon_has_coordinate_types(result, x_type=x_type, y_type=y_type)
    assert polygon_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )
    assert is_polygon_strict(result)
    assert is_contour_non_self_intersecting(result.border)
    assert all(is_contour_non_self_intersecting(hole) for hole in result.holes)
    assert contours_do_not_cross_or_overlap(result.holes)
    assert is_contour_counterclockwise(result.border)
    assert all(not is_contour_counterclockwise(hole) for hole in result.holes)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategies_with_limits_and_types,
    strategies.concave_contours_sizes_pairs,
    strategies.polygon_holes_sizes_pairs,
    strategies.convex_contours_sizes_pairs,
)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
    sizes_pair: SizePair,
    holes_sizes_pair: SizePair,
    hole_sizes_pair: SizePair,
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair
    min_holes_size, max_holes_size = holes_sizes_pair
    min_hole_size, max_hole_size = hole_sizes_pair

    strategy = polygons(
        coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )

    result = data.draw(strategy)

    assert isinstance(result, context.polygon_cls)

    assert polygon_has_valid_sizes(
        result,
        min_size=min_size,
        max_size=max_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )
    assert polygon_has_coordinate_types(result, x_type=type_, y_type=type_)
    assert polygon_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
    assert is_polygon_strict(result)
    assert is_contour_non_self_intersecting(result.border)
    assert all(is_contour_non_self_intersecting(hole) for hole in result.holes)
    assert contours_do_not_cross_or_overlap(result.holes)
    assert is_contour_counterclockwise(result.border)
    assert all(not is_contour_counterclockwise(hole) for hole in result.holes)


@given(
    strategies.scalars_strategies,
    strategies.invalid_convex_contours_sizes_pairs,
)
def test_invalid_border_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        polygons(
            coordinates, context=context, min_size=min_size, max_size=max_size
        )


@given(
    strategies.scalars_strategies, strategies.invalid_polygon_holes_sizes_pairs
)
def test_invalid_holes_list_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_holes_size, max_holes_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        polygons(
            coordinates,
            context=context,
            min_holes_size=min_holes_size,
            max_holes_size=max_holes_size,
        )


@given(
    strategies.scalars_strategies,
    strategies.invalid_convex_contours_sizes_pairs,
)
def test_invalid_holes_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_hole_size, max_hole_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        polygons(
            coordinates,
            context=context,
            min_hole_size=min_hole_size,
            max_hole_size=max_hole_size,
        )


@given(
    strategies.scalars_strategies,
    strategies.non_valid_convex_contours_sizes_pairs,
)
def test_non_valid_border_sizes(
    coordinates: st.SearchStrategy[ScalarT], non_valid_sizes_pair: SizePair
) -> None:
    min_size, max_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        polygons(
            coordinates, context=context, min_size=min_size, max_size=max_size
        )

    assert len(warnings) == 1


@given(
    strategies.scalars_strategies,
    strategies.non_valid_convex_contours_sizes_pairs,
)
def test_non_valid_holes_sizes(
    coordinates: st.SearchStrategy[ScalarT], non_valid_sizes_pair: SizePair
) -> None:
    min_size, max_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        polygons(
            coordinates,
            context=context,
            min_hole_size=min_size,
            max_hole_size=max_size,
        )

    assert len(warnings) == 1
