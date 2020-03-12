from typing import Tuple

from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import bounding_boxes
from hypothesis_geometry.utils import sort_pair
from tests import strategies
from tests.utils import (CoordinatesLimitsType,
                         is_bounding_box)


@given(strategies.coordinates_strategies)
def test_basic(coordinates: Strategy[Coordinate]) -> None:
    result = bounding_boxes(coordinates)

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

    strategy = bounding_boxes(x_coordinates, y_coordinates)

    result = data.draw(strategy)

    assert is_bounding_box(result)
    assert len(result) == 4
    assert all(isinstance(coordinate, x_type)
               for coordinate in result[:2])
    assert all(isinstance(coordinate, y_type)
               for coordinate in result[2:])
    assert all(coordinate >= min_x_value
               and (max_x_value is None or coordinate <= max_x_value)
               for coordinate in result[:2])
    assert all(coordinate >= min_y_value
               and (max_y_value is None or coordinate <= max_y_value)
               for coordinate in result[2:])
    assert sort_pair(result[:2]) == result[:2]
    assert sort_pair(result[2:]) == result[2:]


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: CoordinatesLimitsType
                          ) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = bounding_boxes(coordinates)

    result = data.draw(strategy)

    assert is_bounding_box(result)
    assert len(result) == 4
    assert all(isinstance(coordinate, type_)
               for coordinate in result)
    assert all(coordinate >= min_value
               and (max_value is None or coordinate <= max_value)
               for coordinate in result)
    assert sort_pair(result[:2]) == result[:2]
    assert sort_pair(result[2:]) == result[2:]
