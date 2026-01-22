import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import HypothesisWarning

from hypothesis_geometry.planar import multipolygons
from tests import strategies
from tests.hints import ScalarT
from tests.utils import (
    ScalarStrategyLimitsWithType,
    SizePair,
    context,
    contours_do_not_cross_or_overlap,
    is_contour_counterclockwise,
    is_contour_non_self_intersecting,
    is_multipolygon_strict,
    multipolygon_has_coordinate_types,
    multipolygon_has_coordinates_in_range,
    multipolygon_has_valid_sizes,
)


@given(
    strategies.scalars_strategies,
    strategies.multipolygons_sizes_pairs,
    strategies.concave_contours_sizes_pairs,
    strategies.polygon_holes_sizes_pairs,
    strategies.convex_contours_sizes_pairs,
)
def test_basic(
    coordinates: st.SearchStrategy[ScalarT],
    sizes_pair: SizePair,
    border_sizes_pair: SizePair,
    holes_sizes_pair: SizePair,
    hole_sizes_pair: SizePair,
) -> None:
    min_size, max_size = sizes_pair
    min_border_size, max_border_size = border_sizes_pair
    min_holes_size, max_holes_size = holes_sizes_pair
    min_hole_size, max_hole_size = hole_sizes_pair

    result = multipolygons(
        coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
        min_border_size=min_border_size,
        max_border_size=max_border_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )

    assert isinstance(result, st.SearchStrategy)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategy_with_limit_and_type_pairs,
    strategies.multipolygons_sizes_pairs,
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
    border_sizes_pair: SizePair,
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
    min_border_size, max_border_size = border_sizes_pair
    min_holes_size, max_holes_size = holes_sizes_pair
    min_hole_size, max_hole_size = hole_sizes_pair

    strategy = multipolygons(
        x_coordinates,
        y_coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
        min_border_size=min_border_size,
        max_border_size=max_border_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )

    result = data.draw(strategy)

    assert isinstance(result, context.multipolygon_cls)
    assert multipolygon_has_valid_sizes(
        result,
        min_size=min_size,
        max_size=max_size,
        min_border_size=min_border_size,
        max_border_size=max_border_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )
    assert multipolygon_has_coordinate_types(
        result, x_type=x_type, y_type=y_type
    )
    assert multipolygon_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )
    assert is_multipolygon_strict(result)
    assert all(
        is_contour_non_self_intersecting(polygon.border)
        and all(
            is_contour_non_self_intersecting(hole) for hole in polygon.holes
        )
        for polygon in result.polygons
    )
    assert contours_do_not_cross_or_overlap(
        [polygon.border for polygon in result.polygons]
    )
    assert all(
        contours_do_not_cross_or_overlap(polygon.holes)
        for polygon in result.polygons
    )
    assert all(
        is_contour_counterclockwise(polygon.border)
        and all(
            not is_contour_counterclockwise(hole) for hole in polygon.holes
        )
        for polygon in result.polygons
    )


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategies_with_limits_and_types,
    strategies.multipolygons_sizes_pairs,
    strategies.concave_contours_sizes_pairs,
    strategies.polygon_holes_sizes_pairs,
    strategies.convex_contours_sizes_pairs,
)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
    sizes_pair: SizePair,
    border_sizes_pair: SizePair,
    holes_sizes_pair: SizePair,
    hole_sizes_pair: SizePair,
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair
    min_border_size, max_border_size = border_sizes_pair
    min_holes_size, max_holes_size = holes_sizes_pair
    min_hole_size, max_hole_size = hole_sizes_pair

    strategy = multipolygons(
        coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
        min_border_size=min_border_size,
        max_border_size=max_border_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )

    result = data.draw(strategy)

    assert isinstance(result, context.multipolygon_cls)
    assert multipolygon_has_valid_sizes(
        result,
        min_size=min_size,
        max_size=max_size,
        min_border_size=min_border_size,
        max_border_size=max_border_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )
    assert multipolygon_has_coordinate_types(
        result, x_type=type_, y_type=type_
    )
    assert multipolygon_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
    assert is_multipolygon_strict(result)
    assert all(
        is_contour_non_self_intersecting(polygon.border)
        and all(
            is_contour_non_self_intersecting(hole) for hole in polygon.holes
        )
        for polygon in result.polygons
    )
    assert contours_do_not_cross_or_overlap(
        [polygon.border for polygon in result.polygons]
    )
    assert all(
        contours_do_not_cross_or_overlap(polygon.holes)
        for polygon in result.polygons
    )
    assert all(
        is_contour_counterclockwise(polygon.border)
        and all(
            not is_contour_counterclockwise(hole) for hole in polygon.holes
        )
        for polygon in result.polygons
    )


@given(
    strategies.scalars_strategies, strategies.invalid_multipolygons_sizes_pairs
)
def test_invalid_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        multipolygons(
            coordinates, context=context, min_size=min_size, max_size=max_size
        )


@given(
    strategies.scalars_strategies,
    strategies.invalid_convex_contours_sizes_pairs,
)
def test_invalid_border_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_border_size, max_border_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        multipolygons(
            coordinates,
            context=context,
            min_border_size=min_border_size,
            max_border_size=max_border_size,
        )


@given(
    strategies.scalars_strategies, strategies.invalid_polygon_holes_sizes_pairs
)
def test_invalid_holes_list_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_holes_size, max_holes_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        multipolygons(
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
        multipolygons(
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
    min_border_size, max_border_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        multipolygons(
            coordinates,
            context=context,
            min_border_size=min_border_size,
            max_border_size=max_border_size,
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
        multipolygons(
            coordinates,
            context=context,
            min_hole_size=min_size,
            max_hole_size=max_size,
        )

    assert len(warnings) == 1
