from typing import Tuple

from ground.hints import Scalar
from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import Strategy
from hypothesis_geometry.planar import segments
from tests import strategies
from tests.utils import (ScalarsLimitsType,
                         is_segment,
                         segment_has_coordinates_in_range,
                         segment_has_coordinates_types)


@given(strategies.scalars_strategies)
def test_basic(coordinates: Strategy[Scalar]) -> None:
    result = segments(coordinates)

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

    strategy = segments(x_coordinates, y_coordinates)

    result = data.draw(strategy)

    assert is_segment(result)
    assert segment_has_coordinates_types(result,
                                         x_type=x_type,
                                         y_type=y_type)
    assert segment_has_coordinates_in_range(result,
                                            min_x_value=min_x_value,
                                            max_x_value=max_x_value,
                                            min_y_value=min_y_value,
                                            max_y_value=max_y_value)
    assert result.start != result.end


@given(strategies.data,
       strategies.scalars_strategies_with_limits_and_types)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: ScalarsLimitsType
                          ) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = segments(coordinates)

    result = data.draw(strategy)

    assert is_segment(result)
    assert segment_has_coordinates_types(result,
                                         x_type=type_,
                                         y_type=type_)
    assert segment_has_coordinates_in_range(result,
                                            min_x_value=min_value,
                                            max_x_value=max_value,
                                            min_y_value=min_value,
                                            max_y_value=max_value)
    assert result.start != result.end
