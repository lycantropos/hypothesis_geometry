from typing import Tuple

import pytest
from ground.hints import Scalar
from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import Strategy
from hypothesis_geometry.planar import multisegments
from tests import strategies
from tests.utils import (ScalarsLimitsType,
                         SizesPair,
                         has_valid_size,
                         is_multisegment,
                         multisegment_has_coordinates_in_range,
                         multisegment_has_coordinates_types,
                         segments_do_not_cross_or_overlap)


@given(strategies.scalars_strategies,
       strategies.multisegments_sizes_pairs)
def test_basic(coordinates: Strategy[Scalar],
               sizes_pair: SizesPair) -> None:
    min_size, max_size = sizes_pair

    result = multisegments(coordinates,
                           min_size=min_size,
                           max_size=max_size)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.scalars_strategy_with_limit_and_type_pairs,
       strategies.multisegments_sizes_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[ScalarsLimitsType,
                                                        ScalarsLimitsType],
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
    assert has_valid_size(result.segments,
                          min_size=min_size,
                          max_size=max_size)
    assert multisegment_has_coordinates_types(result, x_type=x_type,
                                              y_type=y_type)
    assert multisegment_has_coordinates_in_range(result,
                                                 min_x_value=min_x_value,
                                                 max_x_value=max_x_value,
                                                 min_y_value=min_y_value,
                                                 max_y_value=max_y_value)
    assert segments_do_not_cross_or_overlap(result.segments)


@given(strategies.data,
       strategies.scalars_strategies_with_limits_and_types,
       strategies.multisegments_sizes_pairs)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: ScalarsLimitsType,
                          sizes_pair: SizesPair) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair

    strategy = multisegments(coordinates,
                             min_size=min_size,
                             max_size=max_size)

    result = data.draw(strategy)

    assert is_multisegment(result)
    assert has_valid_size(result.segments,
                          min_size=min_size,
                          max_size=max_size)
    assert multisegment_has_coordinates_types(result,
                                              x_type=type_,
                                              y_type=type_)
    assert multisegment_has_coordinates_in_range(result,
                                                 min_x_value=min_value,
                                                 max_x_value=max_value,
                                                 min_y_value=min_value,
                                                 max_y_value=max_value)
    assert segments_do_not_cross_or_overlap(result.segments)


@given(strategies.scalars_strategies,
       strategies.invalid_multisegments_sizes_pairs)
def test_invalid_sizes(coordinates: Strategy[Scalar],
                       invalid_sizes_pair: SizesPair) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        multisegments(coordinates,
                      min_size=min_size,
                      max_size=max_size)
