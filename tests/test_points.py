from typing import Tuple

from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import points
from tests import strategies
from tests.utils import (CoordinatesLimitsType,
                         is_point,
                         point_has_coordinates_in_range,
                         point_has_coordinates_types)


@given(strategies.coordinates_strategies)
def test_basic(coordinates: Strategy[Coordinate]) -> None:
    result = points(coordinates)

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

    strategy = points(x_coordinates, y_coordinates)

    result = data.draw(strategy)

    assert is_point(result)
    assert point_has_coordinates_types(result,
                                       x_type=x_type,
                                       y_type=y_type)
    assert point_has_coordinates_in_range(result,
                                          min_x_value=min_x_value,
                                          max_x_value=max_x_value,
                                          min_y_value=min_y_value,
                                          max_y_value=max_y_value)


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: CoordinatesLimitsType
                          ) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = points(coordinates)

    result = data.draw(strategy)

    assert is_point(result)
    assert point_has_coordinates_types(result,
                                       x_type=type_,
                                       y_type=type_)
    assert point_has_coordinates_in_range(result,
                                          min_x_value=min_value,
                                          max_x_value=max_value,
                                          min_y_value=min_value,
                                          max_y_value=max_value)
