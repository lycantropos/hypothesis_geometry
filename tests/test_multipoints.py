from typing import Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import multipoints
from tests import strategies
from tests.utils import (CoordinatesLimitsType,
                         SizesPair,
                         all_unique,
                         has_valid_size,
                         is_multipoint,
                         multipoint_has_coordinates_in_range,
                         multipoint_has_coordinates_types)


@given(strategies.coordinates_strategies,
       strategies.multipoints_sizes_pairs)
def test_basic(coordinates: Strategy[Coordinate],
               sizes_pair: SizesPair) -> None:
    min_size, max_size = sizes_pair

    result = multipoints(coordinates,
                         min_size=min_size,
                         max_size=max_size)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.coordinates_strategy_with_limit_and_type_pairs,
       strategies.multipoints_sizes_pairs)
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

    strategy = multipoints(x_coordinates, y_coordinates,
                           min_size=min_size,
                           max_size=max_size)

    result = data.draw(strategy)

    assert is_multipoint(result)
    assert has_valid_size(result.points,
                          min_size=min_size,
                          max_size=max_size)
    assert multipoint_has_coordinates_types(result,
                                            x_type=x_type,
                                            y_type=y_type)
    assert multipoint_has_coordinates_in_range(result,
                                               min_x_value=min_x_value,
                                               max_x_value=max_x_value,
                                               min_y_value=min_y_value,
                                               max_y_value=max_y_value)
    assert all_unique(result.points)


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types,
       strategies.multipoints_sizes_pairs)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: CoordinatesLimitsType,
                          sizes_pair: SizesPair) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair

    strategy = multipoints(coordinates,
                           min_size=min_size,
                           max_size=max_size)

    result = data.draw(strategy)

    assert is_multipoint(result)
    assert has_valid_size(result.points,
                          min_size=min_size,
                          max_size=max_size)
    assert multipoint_has_coordinates_types(result,
                                            x_type=type_,
                                            y_type=type_)
    assert multipoint_has_coordinates_in_range(result,
                                               min_x_value=min_value,
                                               max_x_value=max_value,
                                               min_y_value=min_value,
                                               max_y_value=max_value)
    assert all_unique(result.points)


@given(strategies.coordinates_strategies,
       strategies.invalid_multipoints_sizes_pairs)
def test_invalid_sizes(coordinates: Strategy[Coordinate],
                       invalid_sizes_pair: SizesPair) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        multipoints(coordinates,
                    min_size=min_size,
                    max_size=max_size)
