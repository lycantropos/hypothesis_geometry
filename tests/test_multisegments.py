from typing import Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import multisegments
from tests import strategies
from tests.utils import (CoordinatesLimitsType, SizesPair, has_valid_size,
                         is_multisegment, segment_has_coordinates_in_range,
                         segment_has_coordinates_types,
                         segments_do_not_cross_or_overlap)


@given(strategies.coordinates_strategies,
       strategies.multisegments_sizes_pairs)
def test_basic(coordinates: Strategy[Coordinate],
               sizes_pair: SizesPair) -> None:
    min_size, max_size = sizes_pair

    result = multisegments(coordinates,
                           min_size=min_size,
                           max_size=max_size)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.coordinates_strategy_with_limit_and_type_pairs,
       strategies.multisegments_sizes_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[CoordinatesLimitsType,
                                                        CoordinatesLimitsType],
                    sizes_pair: SizesPair) -> None:
    (x_coordinates_limits_type,
     y_coordinates_limits_type) = coordinates_limits_type_pair
    ((x_coordinates, (min_x_value, max_x_value)),
     x_type) = x_coordinates_limits_type
    ((y_coordinates, (min_y_value, max_y_value)),
     y_type) = y_coordinates_limits_type
    min_size, max_size = sizes_pair

    strategy = multisegments(x_coordinates, y_coordinates,
                             min_size=min_size,
                             max_size=max_size)

    result = data.draw(strategy)

    assert is_multisegment(result)
    assert has_valid_size(result,
                          min_size=min_size,
                          max_size=max_size)
    assert all(segment_has_coordinates_types(segment,
                                             x_type=x_type,
                                             y_type=y_type)
               for segment in result)
    assert all(segment_has_coordinates_in_range(segment,
                                                min_x_value=min_x_value,
                                                max_x_value=max_x_value,
                                                min_y_value=min_y_value,
                                                max_y_value=max_y_value)
               for segment in result)
    assert segments_do_not_cross_or_overlap(result)


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types,
       strategies.multisegments_sizes_pairs)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: CoordinatesLimitsType,
                          sizes_pair: SizesPair) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair

    strategy = multisegments(coordinates,
                             min_size=min_size,
                             max_size=max_size)

    result = data.draw(strategy)

    assert is_multisegment(result)
    assert has_valid_size(result,
                          min_size=min_size,
                          max_size=max_size)
    assert all(segment_has_coordinates_types(segment,
                                             x_type=type_,
                                             y_type=type_)
               for segment in result)
    assert all(segment_has_coordinates_in_range(segment,
                                                min_x_value=min_value,
                                                max_x_value=max_value,
                                                min_y_value=min_value,
                                                max_y_value=max_value)
               for segment in result)
    assert segments_do_not_cross_or_overlap(result)


@given(strategies.coordinates_strategies,
       strategies.invalid_multisegments_sizes_pairs)
def test_invalid_sizes(coordinates: Strategy[Coordinate],
                       invalid_sizes_pair: SizesPair) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        multisegments(coordinates,
                      min_size=min_size,
                      max_size=max_size)