import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import HypothesisWarning

from hypothesis_geometry.planar import mixes
from tests import strategies
from tests.hints import ScalarT
from tests.utils import (
    ScalarStrategyLimitsWithType,
    SizePair,
    context,
    mix_discrete_component_is_disjoint_with_others,
    mix_has_coordinate_types,
    mix_has_coordinates_in_range,
    mix_has_valid_sizes,
    mix_segments_do_not_cross_or_overlap,
)


@given(
    strategies.scalars_strategies,
    strategies.mix_components_sizes_pairs_triplets,
    strategies.concave_contours_sizes_pairs,
    strategies.polygon_holes_sizes_pairs,
    strategies.convex_contours_sizes_pairs,
)
def test_basic(
    coordinates: st.SearchStrategy[ScalarT],
    components_sizes_pair: tuple[SizePair, SizePair, SizePair],
    polygons_border_sizes_pair: SizePair,
    polygons_holes_list_sizes_pair: SizePair,
    polygons_holes_sizes_pair: SizePair,
) -> None:
    points_sizes_pair, segments_sizes_pair, polygons_sizes_pair = (
        components_sizes_pair
    )
    min_points_size, max_points_size = points_sizes_pair
    min_segments_size, max_segments_size = segments_sizes_pair
    min_polygons_size, max_polygons_size = polygons_sizes_pair
    (min_mix_polygon_border_size, max_mix_polygon_border_size) = (
        polygons_border_sizes_pair
    )
    (min_mix_polygon_holes_size, max_mix_polygon_holes_size) = (
        polygons_holes_list_sizes_pair
    )
    (min_mix_polygon_hole_size, max_mix_polygon_hole_size) = (
        polygons_holes_sizes_pair
    )

    result = mixes(
        coordinates,
        min_points_size=min_points_size,
        max_points_size=max_points_size,
        min_segments_size=min_segments_size,
        max_segments_size=max_segments_size,
        min_polygons_size=min_polygons_size,
        max_polygons_size=max_polygons_size,
        min_polygon_border_size=min_mix_polygon_border_size,
        max_polygon_border_size=max_mix_polygon_border_size,
        min_polygon_holes_size=min_mix_polygon_holes_size,
        max_polygon_holes_size=max_mix_polygon_holes_size,
        min_polygon_hole_size=min_mix_polygon_hole_size,
        max_polygon_hole_size=max_mix_polygon_hole_size,
    )

    assert isinstance(result, st.SearchStrategy)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategy_with_limit_and_type_pairs,
    strategies.mix_components_sizes_pairs_triplets,
    strategies.concave_contours_sizes_pairs,
    strategies.polygon_holes_sizes_pairs,
    strategies.convex_contours_sizes_pairs,
)
def test_properties(
    data: st.DataObject,
    coordinate_limits_type_pair: tuple[
        ScalarStrategyLimitsWithType[ScalarT],
        ScalarStrategyLimitsWithType[ScalarT],
    ],
    component_size_pair: tuple[SizePair, SizePair, SizePair],
    polygon_border_size_pair: SizePair,
    polygon_hole_sequence_size_pair: SizePair,
    polygon_hole_size_pair: SizePair,
) -> None:
    (x_coordinates_limits_type, y_coordinates_limits_type) = (
        coordinate_limits_type_pair
    )
    ((x_coordinates, (min_x_value, max_x_value)), x_type) = (
        x_coordinates_limits_type
    )
    ((y_coordinates, (min_y_value, max_y_value)), y_type) = (
        y_coordinates_limits_type
    )
    points_sizes_pair, segments_sizes_pair, polygons_sizes_pair = (
        component_size_pair
    )
    min_points_size, max_points_size = points_sizes_pair
    min_segments_size, max_segments_size = segments_sizes_pair
    min_polygons_size, max_polygons_size = polygons_sizes_pair
    (min_polygon_border_size, max_polygon_border_size) = (
        polygon_border_size_pair
    )
    min_polygon_holes_size, max_polygon_holes_size = (
        polygon_hole_sequence_size_pair
    )
    min_polygon_hole_size, max_polygon_hole_size = polygon_hole_size_pair

    strategy = mixes(
        x_coordinates,
        y_coordinates,
        min_points_size=min_points_size,
        max_points_size=max_points_size,
        min_segments_size=min_segments_size,
        max_segments_size=max_segments_size,
        min_polygons_size=min_polygons_size,
        max_polygons_size=max_polygons_size,
        min_polygon_border_size=min_polygon_border_size,
        max_polygon_border_size=max_polygon_border_size,
        min_polygon_holes_size=min_polygon_holes_size,
        max_polygon_holes_size=max_polygon_holes_size,
        min_polygon_hole_size=min_polygon_hole_size,
        max_polygon_hole_size=max_polygon_hole_size,
    )

    result = data.draw(strategy)

    assert isinstance(result, context.mix_cls)
    assert mix_has_valid_sizes(
        result,
        min_points_size=min_points_size,
        max_points_size=max_points_size,
        min_segments_size=min_segments_size,
        max_segments_size=max_segments_size,
        min_polygons_size=min_polygons_size,
        max_polygons_size=max_polygons_size,
        min_polygon_border_size=min_polygon_border_size,
        max_polygon_border_size=max_polygon_border_size,
        min_polygon_holes_size=min_polygon_holes_size,
        max_polygon_holes_size=max_polygon_holes_size,
        min_polygon_hole_size=min_polygon_hole_size,
        max_polygon_hole_size=max_polygon_hole_size,
    )
    assert mix_has_coordinate_types(result, x_type=x_type, y_type=y_type)
    assert mix_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )
    assert mix_discrete_component_is_disjoint_with_others(result)
    assert mix_segments_do_not_cross_or_overlap(result)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategies_with_limits_and_types,
    strategies.mix_components_sizes_pairs_triplets,
    strategies.concave_contours_sizes_pairs,
    strategies.polygon_holes_sizes_pairs,
    strategies.convex_contours_sizes_pairs,
)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
    components_sizes_pair: tuple[SizePair, SizePair, SizePair],
    polygons_border_sizes_pair: SizePair,
    polygon_holes_sizes_pair: SizePair,
    polygon_hole_sizes_pair: SizePair,
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    points_sizes_pair, segments_sizes_pair, polygons_sizes_pair = (
        components_sizes_pair
    )
    min_points_size, max_points_size = points_sizes_pair
    min_segments_size, max_segments_size = segments_sizes_pair
    min_polygons_size, max_polygons_size = polygons_sizes_pair
    (min_polygon_border_size, max_polygon_border_size) = (
        polygons_border_sizes_pair
    )
    min_polygon_holes_size, max_polygon_holes_size = polygon_holes_sizes_pair
    min_polygon_hole_size, max_polygon_hole_size = polygon_hole_sizes_pair

    strategy = mixes(
        coordinates,
        min_points_size=min_points_size,
        max_points_size=max_points_size,
        min_segments_size=min_segments_size,
        max_segments_size=max_segments_size,
        min_polygons_size=min_polygons_size,
        max_polygons_size=max_polygons_size,
        min_polygon_border_size=min_polygon_border_size,
        max_polygon_border_size=max_polygon_border_size,
        min_polygon_holes_size=min_polygon_holes_size,
        max_polygon_holes_size=max_polygon_holes_size,
        min_polygon_hole_size=min_polygon_hole_size,
        max_polygon_hole_size=max_polygon_hole_size,
    )

    result = data.draw(strategy)

    assert isinstance(result, context.mix_cls)
    assert mix_has_valid_sizes(
        result,
        min_points_size=min_points_size,
        max_points_size=max_points_size,
        min_segments_size=min_segments_size,
        max_segments_size=max_segments_size,
        min_polygons_size=min_polygons_size,
        max_polygons_size=max_polygons_size,
        min_polygon_border_size=min_polygon_border_size,
        max_polygon_border_size=max_polygon_border_size,
        min_polygon_holes_size=min_polygon_holes_size,
        max_polygon_holes_size=max_polygon_holes_size,
        min_polygon_hole_size=min_polygon_hole_size,
        max_polygon_hole_size=max_polygon_hole_size,
    )
    assert mix_has_coordinate_types(result, x_type=type_, y_type=type_)
    assert mix_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
    assert mix_discrete_component_is_disjoint_with_others(result)
    assert mix_segments_do_not_cross_or_overlap(result)


@given(
    strategies.scalars_strategies,
    strategies.invalid_mix_components_sizes_pairs_triplets,
)
def test_invalid_components_sizes(
    coordinates: st.SearchStrategy[ScalarT],
    invalid_components_sizes_pairs: tuple[SizePair, SizePair, SizePair],
) -> None:
    points_sizes_pair, segments_sizes_pair, polygons_sizes_pair = (
        invalid_components_sizes_pairs
    )
    min_points_size, max_points_size = points_sizes_pair
    min_segments_size, max_segments_size = segments_sizes_pair
    min_polygons_size, max_polygons_size = polygons_sizes_pair

    with pytest.raises(ValueError):
        mixes(
            coordinates,
            min_points_size=min_points_size,
            max_points_size=max_points_size,
            min_segments_size=min_segments_size,
            max_segments_size=max_segments_size,
            min_polygons_size=min_polygons_size,
            max_polygons_size=max_polygons_size,
        )


@given(
    strategies.scalars_strategies, strategies.invalid_mix_points_sizes_pairs
)
def test_invalid_points_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_points_size, max_points_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(
            coordinates,
            min_points_size=min_points_size,
            max_points_size=max_points_size,
        )


@given(
    strategies.scalars_strategies, strategies.invalid_mix_polygons_sizes_pairs
)
def test_invalid_polygons_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_polygons_size, max_polygons_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(
            coordinates,
            min_polygons_size=min_polygons_size,
            max_polygons_size=max_polygons_size,
        )


@given(
    strategies.scalars_strategies, strategies.invalid_mix_segments_sizes_pairs
)
def test_invalid_segments_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_segments_size, max_segments_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(
            coordinates,
            min_segments_size=min_segments_size,
            max_segments_size=max_segments_size,
        )


@given(
    strategies.scalars_strategies,
    strategies.invalid_convex_contours_sizes_pairs,
)
def test_invalid_polygon_border_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_polygon_border_size, max_polygon_border_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(
            coordinates,
            min_polygon_border_size=min_polygon_border_size,
            max_polygon_border_size=max_polygon_border_size,
        )


@given(
    strategies.scalars_strategies, strategies.invalid_polygon_holes_sizes_pairs
)
def test_invalid_polygon_holes_list_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_polygon_holes_size, max_polygon_holes_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(
            coordinates,
            min_polygon_holes_size=min_polygon_holes_size,
            max_polygon_holes_size=max_polygon_holes_size,
        )


@given(
    strategies.scalars_strategies,
    strategies.invalid_convex_contours_sizes_pairs,
)
def test_invalid_polygon_holes_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_polygon_hole_size, max_polygon_hole_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(
            coordinates,
            min_polygon_hole_size=min_polygon_hole_size,
            max_polygon_hole_size=max_polygon_hole_size,
        )


@given(
    strategies.scalars_strategies,
    strategies.non_valid_convex_contours_sizes_pairs,
)
def test_non_valid_polygon_border_sizes(
    coordinates: st.SearchStrategy[ScalarT], non_valid_sizes_pair: SizePair
) -> None:
    min_polygon_border_size, max_polygon_border_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        mixes(
            coordinates,
            min_polygon_border_size=min_polygon_border_size,
            max_polygon_border_size=max_polygon_border_size,
        )

    assert len(warnings) == 1


@given(
    strategies.scalars_strategies,
    strategies.non_valid_convex_contours_sizes_pairs,
)
def test_non_valid_polygon_holes_sizes(
    coordinates: st.SearchStrategy[ScalarT], non_valid_sizes_pair: SizePair
) -> None:
    min_polygon_hole_size, max_polygon_hole_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        mixes(
            coordinates,
            min_polygon_hole_size=min_polygon_hole_size,
            max_polygon_hole_size=max_polygon_hole_size,
        )

    assert len(warnings) == 1
