import pytest
from hypothesis import given, strategies as st

from hypothesis_geometry._core.utils import segments_do_not_cross_or_overlap
from hypothesis_geometry.planar import multisegments
from tests import strategies
from tests.hints import ScalarT
from tests.utils import (
    ScalarStrategyLimitsWithType,
    SizePair,
    context,
    has_valid_size,
    multisegment_has_coordinate_types,
    multisegment_has_coordinates_in_range,
)


@given(strategies.scalars_strategies, strategies.multisegments_sizes_pairs)
def test_basic(
    coordinates: st.SearchStrategy[ScalarT], sizes_pair: SizePair
) -> None:
    min_size, max_size = sizes_pair

    result = multisegments(
        coordinates, context=context, min_size=min_size, max_size=max_size
    )

    assert isinstance(result, st.SearchStrategy)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategy_with_limit_and_type_pairs,
    strategies.multisegments_sizes_pairs,
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

    strategy = multisegments(
        x_coordinates,
        y_coordinates,
        context=context,
        min_size=min_size,
        max_size=max_size,
    )

    result = data.draw(strategy)

    assert isinstance(result, context.multisegment_cls)
    assert has_valid_size(
        result.segments, min_size=min_size, max_size=max_size
    )
    assert multisegment_has_coordinate_types(
        result, x_type=x_type, y_type=y_type
    )
    assert multisegment_has_coordinates_in_range(
        result,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )
    assert segments_do_not_cross_or_overlap(result.segments, context=context)


@given(
    strategies.data_object_strategy,
    strategies.scalars_strategies_with_limits_and_types,
    strategies.multisegments_sizes_pairs,
)
def test_same_coordinates(
    data: st.DataObject,
    coordinates_limits_type: ScalarStrategyLimitsWithType[ScalarT],
    sizes_pair: SizePair,
) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair

    strategy = multisegments(
        coordinates, context=context, min_size=min_size, max_size=max_size
    )

    result = data.draw(strategy)

    assert isinstance(result, context.multisegment_cls)
    assert has_valid_size(
        result.segments, min_size=min_size, max_size=max_size
    )
    assert multisegment_has_coordinate_types(
        result, x_type=type_, y_type=type_
    )
    assert multisegment_has_coordinates_in_range(
        result,
        min_x_value=min_value,
        max_x_value=max_value,
        min_y_value=min_value,
        max_y_value=max_value,
    )
    assert segments_do_not_cross_or_overlap(result.segments, context=context)


@given(
    strategies.scalars_strategies, strategies.invalid_multisegments_sizes_pairs
)
def test_invalid_sizes(
    coordinates: st.SearchStrategy[ScalarT], invalid_sizes_pair: SizePair
) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        multisegments(
            coordinates, context=context, min_size=min_size, max_size=max_size
        )
