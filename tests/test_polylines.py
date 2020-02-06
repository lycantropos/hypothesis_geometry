from typing import Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import polylines
from tests import strategies
from tests.utils import (CoordinatesLimitsType,
                         SizesPair,
                         has_no_consecutive_repetitions,
                         has_valid_size,
                         point_has_coordinates_in_range,
                         point_has_coordinates_types,
                         point_has_valid_size)


@given(strategies.coordinates_strategies, strategies.polylines_sizes_pairs)
def test_basic(coordinates: Strategy[Coordinate],
               sizes_pair: SizesPair) -> None:
    min_size, max_size = sizes_pair

    result = polylines(coordinates,
                       min_size=min_size,
                       max_size=max_size)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.coordinates_strategy_with_limit_and_type_pairs,
       strategies.polylines_sizes_pairs)
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

    strategy = polylines(x_coordinates, y_coordinates,
                         min_size=min_size,
                         max_size=max_size)

    result = data.draw(strategy)

    assert isinstance(result, list)
    assert has_valid_size(result,
                          min_size=min_size,
                          max_size=max_size)
    assert all(isinstance(vertex, tuple) for vertex in result)
    assert all(point_has_valid_size(vertex) for vertex in result)
    assert all(point_has_coordinates_types(vertex,
                                           x_type=x_type,
                                           y_type=y_type)
               for vertex in result)
    assert all(point_has_coordinates_in_range(vertex,
                                              min_x_value=min_x_value,
                                              max_x_value=max_x_value,
                                              min_y_value=min_y_value,
                                              max_y_value=max_y_value)
               for vertex in result)
    assert has_no_consecutive_repetitions(result)


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types,
       strategies.polylines_sizes_pairs)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: CoordinatesLimitsType,
                          sizes_pair: SizesPair) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair

    strategy = polylines(coordinates,
                         min_size=min_size,
                         max_size=max_size)

    result = data.draw(strategy)

    assert isinstance(result, list)
    assert has_valid_size(result,
                          min_size=min_size,
                          max_size=max_size)
    assert all(isinstance(vertex, tuple) for vertex in result)
    assert all(point_has_valid_size(vertex) for vertex in result)
    assert all(point_has_coordinates_types(vertex,
                                           x_type=type_,
                                           y_type=type_)
               for vertex in result)
    assert all(point_has_coordinates_in_range(vertex,
                                              min_x_value=min_value,
                                              max_x_value=max_value,
                                              min_y_value=min_value,
                                              max_y_value=max_value)
               for vertex in result)
    assert has_no_consecutive_repetitions(result)


@given(strategies.coordinates_strategies,
       strategies.invalid_polylines_sizes_pairs)
def test_invalid_sizes(coordinates: Strategy[Coordinate],
                       invalid_sizes_pair: SizesPair) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        polylines(coordinates,
                  min_size=min_size,
                  max_size=max_size)