from typing import Tuple

from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import segments
from tests import strategies
from tests.utils import (CoordinatesLimitsType,
                         all_unique,
                         point_has_coordinates_in_range,
                         point_has_coordinates_types,
                         point_has_valid_size)


@given(strategies.coordinates_strategies)
def test_basic(coordinates: Strategy[Coordinate]) -> None:
    result = segments(coordinates)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.coordinates_strategy_with_limit_and_type_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[CoordinatesLimitsType,
                                                        CoordinatesLimitsType]
                    ) -> None:
    (x_coordinates_limits_type,
     y_coordinates_limits_type) = coordinates_limits_type_pair
    ((x_coordinates, (min_x_value, max_x_value)),
     x_type) = x_coordinates_limits_type
    ((y_coordinates, (min_y_value, max_y_value)),
     y_type) = y_coordinates_limits_type

    strategy = segments(x_coordinates, y_coordinates)

    result = data.draw(strategy)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(endpoint, tuple) for endpoint in result)
    assert all(point_has_valid_size(endpoint) for endpoint in result)
    assert all(point_has_coordinates_types(endpoint,
                                           x_type=x_type,
                                           y_type=y_type)
               for endpoint in result)
    assert all(point_has_coordinates_in_range(endpoint,
                                              min_x_value=min_x_value,
                                              max_x_value=max_x_value,
                                              min_y_value=min_y_value,
                                              max_y_value=max_y_value)
               for endpoint in result)
    assert all_unique(result)


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: CoordinatesLimitsType
                          ) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = segments(coordinates)

    result = data.draw(strategy)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(endpoint, tuple) for endpoint in result)
    assert all(point_has_valid_size(endpoint) for endpoint in result)
    assert all(point_has_coordinates_types(endpoint,
                                           x_type=type_,
                                           y_type=type_)
               for endpoint in result)
    assert all(point_has_coordinates_in_range(endpoint,
                                              min_x_value=min_value,
                                              max_x_value=max_value,
                                              min_y_value=min_value,
                                              max_y_value=max_value)
               for endpoint in result)
    assert all_unique(result)
