from typing import Tuple

from ground.hints import Scalar
from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import Strategy
from hypothesis_geometry.planar import boxes
from tests import strategies
from tests.utils import (ScalarsLimitsType,
                         box_has_coordinates_in_range,
                         box_has_coordinates_types,
                         is_box)


@given(strategies.scalars_strategies)
def test_basic(coordinates: Strategy[Scalar]) -> None:
    result = boxes(coordinates)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.scalars_strategy_with_limit_and_type_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[ScalarsLimitsType,
                                                        ScalarsLimitsType]
                    ) -> None:
    (x_coordinates_limits_type,
     y_coordinates_limits_type) = coordinates_limits_type_pair
    ((x_coordinates, (min_x_value, max_x_value)),
     x_type) = x_coordinates_limits_type
    ((y_coordinates, (min_y_value, max_y_value)),
     y_type) = y_coordinates_limits_type

    strategy = boxes(x_coordinates, y_coordinates)

    result = data.draw(strategy)

    assert is_box(result)
    assert box_has_coordinates_types(result,
                                     x_type=x_type,
                                     y_type=y_type)
    assert box_has_coordinates_in_range(result,
                                        min_x_value=min_x_value,
                                        max_x_value=max_x_value,
                                        min_y_value=min_y_value,
                                        max_y_value=max_y_value)
    assert result.min_x < result.max_x
    assert result.min_y < result.max_y


@given(strategies.data,
       strategies.scalars_strategies_with_limits_and_types)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: ScalarsLimitsType
                          ) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = boxes(coordinates)

    result = data.draw(strategy)

    assert is_box(result)
    assert box_has_coordinates_types(result,
                                     x_type=type_,
                                     y_type=type_)
    assert box_has_coordinates_in_range(result,
                                        min_x_value=min_value,
                                        max_x_value=max_value,
                                        min_y_value=min_value,
                                        max_y_value=max_value)
    assert result.min_x < result.max_x
    assert result.min_y < result.max_y
